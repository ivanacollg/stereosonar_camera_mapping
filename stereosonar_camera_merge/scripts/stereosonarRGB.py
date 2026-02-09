from CFAR import * 
from scipy.interpolate import interp1d
from tf.transformations import euler_from_quaternion, euler_matrix
import rospy
import sys
import cv2
from sklearn.utils import shuffle

class StereoSonarRGB:
    """A class to handle the multi sonar array"""

    def __init__(self, ns="~"):
        # type: (str) -> None

        # define vertical transformation between sonars (meters)
        self.transformation = 0.1

        # define the implementation we want used
        self.method = 'python'

        #are we going to publish the feature images?
        self.vis_features = True

        # define the max uncertainty for a match
        self.uncertaintyMax = .97
        self.patchSize = 5

        # define the footprint of the sonars, note we assume that theses parameters
        # are SHARED across sonars (the same)
        self.horizontalFOV = 130.0
        self.verticalAperture = 30.0

        # horizontal sonar info
        self.maxRange_horizontal = 3.0
        # vertical sonar info
        self.maxRange_vertical = 3.0  # default value, reads in new value from msg

        # CFAR parameters for horizontal sonar
        self.tcHorizontal = 40
        self.gcHorizontal = 10
        self.pfaHorizontal = .1
        self.thresholdHorizontal = 60

        # CFAR parameters for vertical sonar
        self.tcVertical = 40
        self.gcVertical = 10
        self.pfaVertical = .1
        self.thresholdVertical = 60

        # define the CFAR detectors
        self.detector_horizontal = CFAR(
            self.tcHorizontal, self.gcHorizontal, self.pfaHorizontal, None
        )
        self.detector_vertical = CFAR(
            self.tcVertical, self.gcVertical, self.pfaVertical, None
        )
        
        # for remapping from polar to cartisian
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1
        self.fast_performance = True

        self.id = 0
        self.currentHoriz = np.zeros(0)
        self.currentVert = np.zeros(0)
        self.currentStamp = None
        self.pingmsg = None
        self.new = True

        self.odom = None
        self.odom_new = None
        self.aggregatedCloud = np.zeros(0)


    def generate_map_xy(self, ping):
        # type: (OculusPing) -> None
        """Generate a mesh grid map for the sonar image.

        Keyword Parameters:
        ping -- a OculusPing message
        """

        # get the parameters from the ping message
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = (
            np.sin(self.to_rad(ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        )
        _cols = int(np.ceil(_width / _res))

        # check if the parameters have changed
        if (
            self.res == _res
            and self.height == _height
            and self.rows == _rows
            and self.width == _width
            and self.cols == _cols
        ):
            return

        # if they have changed do some work
        self.res, self.height, self.rows, self.width, self.cols = (
            _res,
            _height,
            _rows,
            _width,
            _cols,
        )

        # generate the mapping
        bearings = self.to_rad(np.asarray(ping.bearings, dtype=np.float32))
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind="linear",
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True,
        )

        # build the meshgrid
        XX, YY = np.meshgrid(range(self.cols), range(self.rows))
        x = self.res * (self.rows - YY)
        y = self.res * (-self.cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x) * self.REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        self.map_y = np.asarray(r / self.res, dtype=np.float32)
        self.map_x = np.asarray(f_bearings(b), dtype=np.float32)
        #np.save("mapy", self.map_y)
        #np.save("mapx", self.map_x)


        # check for change in max range
        if self.maxRange_horizontal != self.height:
            self.maxRange_horizontal = self.height
            self.maxRange_vertical = self.height

    def img2Real_overlaping(self, points, sonar):
        # type: (np.ndarray, str) -> typing.Tuple[np.ndarray, np.ndarray, float, float, float, float]
        """Accepts CFAR points and sonar type, filters based on vertical apature.

        Keyword Parameters:
        points -- pixels from CFAR
        sonar -- sonar type (vertical or horizontal)

        Returns:
        pixels, x, y, range (in meters) and bearing in degrees
        """

        # convert to meters
        x = points[:, 1] - self.cols / 2.0
        x = 1 * ((x / float(self.cols / 2.0)) * (self.width / 2.0))
        y = (-1 * (points[:, 0] / float(self.rows)) * self.height) + self.height

        # check the sonar type
        if sonar == "vertical":
            x -= self.transformation
        elif sonar == "horizontal":
            pass
        else:
            rospy.loginfo("Incorrect sonar info in img2real function!")

        # get range and bearing
        r = np.sqrt(x ** 2 + y ** 2)
        b = np.degrees(np.arctan(x / y))

        # div vertical aperture by two
        deg = self.verticalAperture / 2.0

        # filter based on the vertical apature of the companion sonar
        x = x[(b > -deg) & (b < deg)]  # x meters
        y = y[(b > -deg) & (b < deg)]  # y meters
        r = r[(b > -deg) & (b < deg)]  # range meters
        u = points[:, 0][(b > -deg) & (b < deg)]  # u pixel coords
        v = points[:, 1][(b > -deg) & (b < deg)]  # v pixel coords
        b = b[(b > -deg) & (b < deg)]  # bearing in degrees

        return u, v, x, y, r, b
    
    def img2Real(self, points, sonar):
        # type: (np.ndarray, str) -> typing.Tuple[np.ndarray, np.ndarray, float, float, float, float]
        """Accepts CFAR points and sonar type, filters based on vertical apature.

        Keyword Parameters:
        points -- pixels from CFAR
        sonar -- sonar type (vertical or horizontal)

        Returns:
        pixels, x, y, range (in meters) and bearing in degrees
        """

        # convert to meters
        x = points[:, 1] - self.cols / 2.0
        x = 1 * ((x / float(self.cols / 2.0)) * (self.width / 2.0))
        y = (-1 * (points[:, 0] / float(self.rows)) * self.height) + self.height

        # check the sonar type
        if sonar == "vertical":
            x -= self.transformation
        elif sonar == "horizontal":
            pass
        else:
            rospy.loginfo("Incorrect sonar info in img2real function!")

        # get range and bearing
        r = np.sqrt(x ** 2 + y ** 2)
        b = np.degrees(np.arctan(x / y))

        # div vertical aperture by two
        deg = self.verticalAperture / 2.0

        # filter based on the vertical apature of the companion sonar
        u = points[:, 0]  # u pixel coords
        v = points[:, 1]  # v pixel coords

        return u, v, x, y, r, b


    def extractFeatures(self, img, detector, alg, threshold):
        # type: (np.ndarray, CFAR, str, float) -> np.ndarray
        """Function to take a raw greyscale sonar image and apply CFAR.

        Keyword Parameters:
        img -- raw greyscale sonar image
        detector -- detector object
        alg -- CFAR version to be used
        threshold -- CFAR thershold

        Returns:
        CFAR points as ndarray
        """

        if detector == "vertical":
            detector = self.detector_vertical
        elif detector == "horizontal":
            detector = self.detector_horizontal
        else:
            rospy.loginfo("Incorrect sonar info in extractFeatures function!")

        # get raw detections
        peaks = detector.detect(img, alg)

        # check against threhold
        peaks &= img > threshold

        # convert to cartisian
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)

        # compile points
        points = np.c_[np.nonzero(peaks)]

        # return a numpy array
        return np.array(points), np.array(peaks)

    def extractPatches(self, v, u, sonar, img, size, ravel=False):
        """A function to get the patch around all features.

        Keyword Arguments:
        v, u -- image pixel coords for features
        sonar -- which sonar (vertical or horizontal)
        img -- the raw sonar image in cartisian format
        size -- the size of the image patch sizexsize

        Returns:
        python list of size x size patches around each feature
        """

        # container for the image patches
        patches = []

        # create a padded image to make the kernel always inside the image
        paddedImg = cv2.copyMakeBorder(
            img, size, size, size, size, cv2.BORDER_CONSTANT, value=0
        )

        # loop over the set features
        for j, i in zip(v, u):

            j = j + size
            i = i + size

            # add the kernel to a patch list, if vertical sonar, rotate the patch along the way
            if sonar == "horizontal":
                patch = paddedImg[i - size : 1 + i + size, j - size : 1 + j + size]
                if patch.shape == (size * 2 + 1, size * 2 + 1):
                    if ravel:
                        patch = np.ravel(patch)
                    patches.append(patch)
            elif sonar == "vertical":
                patch = np.rot90(
                    paddedImg[i - size : 1 + i + size, j - size : 1 + j + size], -1
                )
                if patch.shape == (size * 2 + 1, size * 2 + 1):
                    if ravel:
                        patch = np.ravel(patch)
                    patches.append(patch)
            else:
                rospy.loginfo("Incorrect sonar info in extractPatches function!")

        return patches

    def matchFeatures_2(
        self,
        rangeHorizontal,  # type: float
        bearingHorizontal,  # type: float
        patchesHorizontal,  # type: np.ndarray
        rangeVertical,  # type: float
        xVertical,  # type: float
        patchesVertical,  # type: np.ndarray
    ):

        # FIX ME, make the image size a ros param
        # convert range (meters) to pixels
        rangeHorizontal_discret = np.round(600 * (rangeHorizontal / self.maxRange_horizontal))
        rangeVertical_discret = np.round(600 * (rangeVertical / self.maxRange_vertical))

        # call the cpp function to do the matching
        matches =  match_features_cpp(rangeHorizontal_discret,
                                    rangeHorizontal,
                                    bearingHorizontal,
                                    rangeVertical_discret,
                                    rangeVertical,
                                    xVertical,
                                    patchesHorizontal,
                                    patchesVertical)

        return matches, matches.shape != (0,5)


    def matchFeatures(
        self,
        rangeHorizontal,  # type: float
        bearingHorizontal,  # type: float
        xHorizontal,  # type: float
        yHorizontal,  # type: float
        patchesHorizontal,  # type: np.ndarray
        rangeVertical,  # type: float
        bearingVertical,  # type: float
        xVertical,  # type: float
        yVertical,  # type: float
        patchesVertical,  # type: np.ndarray
        uHorizontal,
        vHorizontal,
        uVertical,
        vVertical
    ):
        # type: (...) -> np.ndarray
        """Perform feature matching on sub problems.

        Keyword Parameters:
        rangeHorizontal -- horizontal sonar range meas (meters)
        bearingHorizontal -- horizontal sonar bearing meas (degrees)
        xHorizontal, yHorizontal -- horizontal sonar meas (meters) in cartisian
        patchesHorizontal -- image patches from horizontal sonar meas

        rangeVertical -- vertical sonar range meas (meters)
        bearingVertical -- vertical sonar bearing meas (degrees)
        xVertical, yVertical -- vertical sonar meas (meters) in cartisian
        patchesVertical -- image patches from vertical sonar meas

        Returns:
        array of matched features
        """

        # FIX ME, make the image size a ros param
        # convert range (meters) to pixels
        rangeHorizontal_discret = np.round(
            600 * (rangeHorizontal / self.maxRange_horizontal)
        )
        rangeVertical_discret = np.round(600 * (rangeVertical / self.maxRange_vertical))
        #np.save("horizontal_ranges", rangeHorizontal_discret)
        #np.save("vertical_ranges", rangeVertical_discret)
        #np.save("horizontal_ranges2", rangeHorizontal)
        #np.save("vertical_ranges2", rangeVertical)

        # get the unique range options
        range_options_horizontal = np.sort(list(set(rangeHorizontal_discret)))
        range_options_vertical = np.sort(list(set(rangeVertical_discret)))
        #np.save("horizontal_ranges_opt2", range_options_horizontal)
        #np.save("vertical_ranges_opt2", range_options_vertical)

        # create some convient containers
        featuresHorizontal = np.column_stack((rangeHorizontal, bearingHorizontal))
        featuresVertical = np.column_stack((xVertical, rangeVertical))

        #pixel containers
        pixelsHorizontal = np.column_stack((uHorizontal, vHorizontal))
        pixelsVertical = np.column_stack((uVertical, vVertical))

        # container for matches format: [x,y,y,z,uncer]
        matches = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0])
        closest_range = np.inf
        close_flag = True
        # loop over the range options in horizontal sonar
        for r1 in range_options_horizontal:

            # check range options in vertical sonar
            if (r1 in range_options_vertical) == True:
                # get features at this range
                hor = featuresHorizontal[rangeHorizontal_discret == r1]
                vert = featuresVertical[rangeVertical_discret == r1]

                # get the patches at this range
                hor_kernel = patchesHorizontal[rangeHorizontal_discret == r1]
                vert_kernel = patchesVertical[rangeVertical_discret == r1]

                # get pixels at this range
                hor_pix = pixelsHorizontal[rangeHorizontal_discret == r1]
                vert_pix = pixelsVertical[rangeVertical_discret == r1]

                # guard the sub problem against noise
                if len(hor) > 2 and len(vert) > 2:
                    
                    # Cartesian product
                    hor_repeat = np.repeat(hor, len(vert), axis=0)      # Shape: (m*n, 2)
                    hor_pix_repeat = np.repeat(hor_pix, len(vert), axis=0)
                    vert_tile = np.tile(vert, (len(hor), 1))             # Shape: (m*n, 2)
                    vert_pix_tile = np.tile(vert_pix, (len(hor),1))
                    # Concatenate along columns
                    matches = np.row_stack(
                            (
                                matches,
                                np.hstack((hor_repeat, vert_tile, hor_pix_repeat, vert_pix_tile))
                                ,
                            )
                        )
                    
                    if close_flag: 
                        closest_range = np.min(hor[:, 0]) # minimum range in the horizontal sonar that has a vertical pair
                        close_flag = False
                

        return matches, matches.shape != (8,), closest_range


    def getClosestPoints(self, rangeHorizontal, bearingHorizontal, uh, vh, xh, yh, closest_match):
        points =np.column_stack((rangeHorizontal, bearingHorizontal, uh, vh, xh, yh))
        filtered_points = points[points[:, 0] < closest_match]
        
        return filtered_points
    
    
    def run_stereo(self, imgHorizontal, imgVertical, stamp, ping):

        # package as numpy array
        stereo_pointcloud = np.column_stack(([], [], []))
        close_pointcloud_h = np.column_stack (([], [], []))
        close_points_h = np.column_stack (([], [], [], [], [], []))
        close_pointcloud_v = np.column_stack (([], [], []))
        close_points_v = np.column_stack (([], [], [], [], [], []))


        if imgHorizontal.size > 0 and imgVertical.size > 0 and ping is not None: 
            self.new = False
            # generate the mapping from polar to cartisian
            self.generate_map_xy(ping)

            # denoise the horizontal image, consider adding this for the vertical image
            if not self.fast_performance:
                imgHorizontal = cv2.fastNlMeansDenoising(imgHorizontal, None, 10, 7, 21)
                imgVertical = cv2.fastNlMeansDenoising(imgVertical, None, 10, 7, 21)

            # check size, images must be the same size
            if imgHorizontal.shape != imgVertical.shape:
                imgVertical = cv2.resize(
                    imgVertical, (imgHorizontal.shape[1], imgHorizontal.shape[0])
                )

            # get some features using CFAR
            horizontalFeatures, horizontalFeatureImage = self.extractFeatures(
                imgHorizontal, "horizontal", "SOCA", self.thresholdHorizontal
            )
            verticalFeatures, verticalFeatureImage = self.extractFeatures(
                imgVertical, "vertical", "SOCA", self.thresholdVertical
            )

            # publish the CFAR image
            if self.vis_features:
                horizontalFeatureImage *= 255
                verticalFeatureImage *= 255
                # Visualize FOV
                horizFeatureImage = cv2.cvtColor(horizontalFeatureImage,cv2.COLOR_GRAY2RGB)
                vertFeatureImage = cv2.cvtColor(verticalFeatureImage,cv2.COLOR_GRAY2RGB)     
                cv2.line(horizFeatureImage,(int(horizontalFeatureImage.shape[1]/2 - horizontalFeatureImage.shape[0]*np.sin(np.deg2rad(15))),0),(int(horizontalFeatureImage.shape[1]/2),horizontalFeatureImage.shape[0]),[0,0,255],5)
                cv2.line(horizFeatureImage,(int(horizontalFeatureImage.shape[1]/2 + horizontalFeatureImage.shape[0]*np.sin(np.deg2rad(15))),0),(int(horizontalFeatureImage.shape[1]/2),horizontalFeatureImage.shape[0]),[0,0,255],5)
                cv2.line(vertFeatureImage,(int(horizontalFeatureImage.shape[1]/2 - horizontalFeatureImage.shape[0]*np.sin(np.deg2rad(15))),0), (int(horizontalFeatureImage.shape[1]/2),horizontalFeatureImage.shape[0]),[0,0,255],5)
                cv2.line(vertFeatureImage,(int(horizontalFeatureImage.shape[1]/2 + horizontalFeatureImage.shape[0]*np.sin(np.deg2rad(15))),0), (int(horizontalFeatureImage.shape[1]/2),horizontalFeatureImage.shape[0]),[0,0,255],5)   


            # remap the raw images into cartisian coords
            imgHorizontal = cv2.remap(
                imgHorizontal, self.map_x, self.map_y, cv2.INTER_LINEAR
            )
            imgVertical = cv2.remap(imgVertical, self.map_x, self.map_y, cv2.INTER_LINEAR)

            # convert the features to meters and degrees
            uh, vh, xh, yh, rh, bh = self.img2Real_overlaping(horizontalFeatures, "horizontal")
            uv, vv, xv, yv, rv, bv = self.img2Real_overlaping(verticalFeatures, "vertical")

            if self.method == "python":

                # get the image kernels, used to compare pixel similarity
                patches_horizontal = np.array(
                    self.extractPatches(vh, uh, "horizontal", imgHorizontal, self.patchSize)
                )
                patches_vertical = np.array(
                    self.extractPatches(vv, uv, "vertical", imgVertical, self.patchSize)
                )

                # perform some matching
                matches, match_status, closest_range = self.matchFeatures(
                    rh, bh, xh, yh, patches_horizontal, rv, bv, xv, yv, patches_vertical, uh, vh, uv, vv
                )

                # remove the first row of zeros, only in the python implmentation
                matches = np.delete(matches, 0, 0)
                
                uh, vh, xh, yh, rh, bh = self.img2Real(horizontalFeatures, "horizontal")
                close_points_h = self.getClosestPoints(rh, bh, uh, vh, xh, yh, closest_range) # return range and bearing of close points
                #print("got close points horizontal")
                uv, vv, xv, yv, rv, bv = self.img2Real(verticalFeatures, "vertical")
                close_points_v = self.getClosestPoints(rv, bv, uv, vv, xv, yv, closest_range) # return range and bearing of close points
                #print("got close points vertical")
                
            elif self.method == "cpp":

                # get the image kernels, used to compare pixel similarity
                patches_horizontal = np.array(
                    self.extractPatches(vh, uh, "horizontal", imgHorizontal, self.patchSize,True)
                )
                patches_vertical = np.array(
                    self.extractPatches(vv, uv, "vertical", imgVertical, self.patchSize,True)
                )

                # perform some matching
                matches, match_status = self.matchFeatures_2(
                    rh, bh, patches_horizontal, rv, xv, patches_vertical
                )


            # protect for no matches
            if match_status:
                # solve the conversion to cartsiain coords from spherical
                bearingAngle = np.radians(matches[:, 1])  # convert back to radians
                rangeAvg = (
                    matches[:, 0] + matches[:, 3]
                ) / 2.0  # average the range between the two matches
                elevationAngle = np.arcsin(
                    matches[:, 2] / rangeAvg
                )  # get the elevation angle
                x = (
                    rangeAvg * np.cos(bearingAngle) * np.cos(elevationAngle)
                )  # convert to cartisian
                y = rangeAvg * np.sin(bearingAngle) * np.cos(elevationAngle)
                z = -matches[:, 2]

                # assemble the point cloud for ROS, the order may be different based on your
                # coordinate frame
                # With Kalman Filter
                stereo_pointcloud = np.column_stack((x, -y, z)) ## Offset to robot center
                
                # publish the CFAR image
                if self.vis_features and matches[:,7].any():
                    for i in range(matches[:,4].shape[0]):
                        cv2.circle(horizFeatureImage,(matches[i,5].astype(int),matches[i,4].astype(int)), 3, (0,255,0), -1)
                        cv2.circle(vertFeatureImage,(matches[i,7].astype(int),matches[i,6].astype(int)), 3, (0, 255, 0), -1)      

            # protect for no matches
            if close_points_h[:,4].any():

                # solve the conversion to cartsiain coords from spherical
                bearingAngle = np.radians(close_points_h[:, 1])  # convert back to radians
                rangeAvg = close_points_h[:, 0]  # average the range between the two matches
                elevationAngle = 0.0  # get the elevation angle
                x = (
                    rangeAvg * np.cos(bearingAngle) * np.cos(elevationAngle)
                )  # convert to cartisian
                y = rangeAvg * np.sin(bearingAngle) * np.cos(elevationAngle)
                z = np.zeros_like(close_points_h[:, 0])
                # With Kalman Filter
                close_pointcloud_h = np.column_stack((x, -y, z)) ## Offset to robot center
                
                if self.vis_features and close_points_h[:,4].any():
                    for i in range(close_points_h[:,4].shape[0]):
                        cv2.circle(horizFeatureImage,(close_points_h[i,3].astype(int),close_points_h[i,2].astype(int)), 3, (255,0,0), -1)
            
            # protect for no matches
            if close_points_v[:,4].any():

                # solve the conversion to cartsiain coords from spherical
                bearingAngle = np.radians(close_points_v[:, 1])  # convert back to radians
                rangeAvg = close_points_v[:, 0]  # average the range between the two matches
                elevationAngle = 0.0  # get the elevation angle
                x = (
                    rangeAvg * np.cos(bearingAngle) * np.cos(elevationAngle)
                )  # convert to cartisian
                y = rangeAvg * np.sin(bearingAngle) * np.cos(elevationAngle)
                z = np.zeros_like(close_points_v[:, 0])
                # With Kalman Filter
                close_pointcloud_v = np.column_stack((x, -y, z)) ## Offset to robot center
                
                if self.vis_features and close_points_v[:,4].any():
                    for i in range(close_points_v[:,4].shape[0]):
                        cv2.circle(vertFeatureImage,(close_points_v[i,3].astype(int),close_points_v[i,2].astype(int)), 3, (255,0,0), -1)
            
            return stereo_pointcloud, horizFeatureImage, vertFeatureImage, close_pointcloud_h, close_points_h, close_pointcloud_v, close_points_v
