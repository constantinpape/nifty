#pragma once
#include <boost/container/flat_map.hpp>

namespace nifty {
namespace container{

    template<class KEY, class VALUE>
    using BoostFlatMap = boost::container::flat_map<KEY, VALUE>;

} // container
} // namespace nifty

