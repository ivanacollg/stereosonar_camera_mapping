#ifndef GPCOCTOMAP_STATE_H
#define GPCOCTOMAP_STATE_H

namespace gpcoctomap {

    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN, UNCERTAIN; after pruning: PRUNED
    enum class State : char {
        FREE,
        OCCUPIED,
        UNKNOWN,
        UNCERTAIN,
        PRUNED
    };

}

#endif // GPCOCTOMAP_STATE_H