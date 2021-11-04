#pragma once
#include <boost/container/flat_set.hpp>

namespace nifty {
namespace container{

    template<class T>
    using BoostFlatSet = boost::container::flat_set<T>;

} // container
} // namespace nifty
