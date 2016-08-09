#pragma once
#ifndef NIFTY_PYTHON_CONVERTER_HXX
#define NIFTY_PYTHON_CONVERTER_HXX

#include <type_traits>
#include <initializer_list>
//#include <pybind11/stl.h>



#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>

#include <nifty/marray/marray.hxx>
#include <nifty/tools/block_access.hxx>

namespace py = pybind11;


namespace nifty{

    struct MyNone{
    };

    inline std::string lowerFirst(const std::string & name){
        auto r = name;
        r[0] = std::tolower(name[0]);
        return r;
    }
}




namespace pybind11
{
    namespace detail
    {
        template <typename Type, size_t DIM> 
        struct pymarray_caster;
    }
}

namespace nifty
{
namespace marray
{

    template <typename Type, size_t DIM = 0> 
    class PyView : public View<Type, false>
    {
        friend struct pybind11::detail::pymarray_caster<Type,DIM>;

      private:
        pybind11::array_t<Type> py_array;

      public:
        template <class ShapeIterator>
        PyView(pybind11::array_t<Type> array, Type *data, ShapeIterator begin, ShapeIterator end)
            : View<Type, false>(begin, end, data, FirstMajorOrder, FirstMajorOrder), py_array(array)
        {
            auto info = py_array.request();
            Type *ptr = (Type *)info.ptr;

            std::vector<size_t> strides(info.strides.begin(),info.strides.end());
            for(size_t i=0; i<strides.size(); ++i){
                strides[i] /= sizeof(Type);
            }
            this->assign( info.shape.begin(), info.shape.end(), strides.begin(), ptr, FirstMajorOrder);

        }

        PyView(std::nullptr_t){

        }
        PyView(MyNone none){

        }

        PyView()
        {
        }
        const Type & operator[](const uint64_t index)const{
            return this->operator()(index);
        }
        Type & operator[](const uint64_t index){
            return this->operator()(index);
        }




        template <class ShapeIterator>
        PyView(ShapeIterator begin, ShapeIterator end)
        {
            this->assignFromShape(begin, end);
        }

        template <class ShapeIterator>
        void reshapeIfEmpty(ShapeIterator begin, ShapeIterator end){
            if(this->size() == 0){
                this->assignFromShape(begin, end);
            }
            else{
                auto c = 0;
                while(begin!=end){
                    if(this->shape(c)!=*begin){
                        throw std::runtime_error("given numpy array has an unusable shape");
                    }
                    ++begin;
                    ++c;
                }
            }
        }



    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        template<class T_INIT>
        PyView(std::initializer_list<T_INIT> shape) : PyView(shape.begin(), shape.end())
        {
        }

        template<class T_INIT>
        void reshapeIfEmpty(std::initializer_list<T_INIT> shape) {
            this->reshapeIfEmpty(shape.begin(), shape.end());
        }
    #endif
    private:

        template <class ShapeIterator>
        void assignFromShape(ShapeIterator begin, ShapeIterator end)
        {
            std::vector<size_t> shape, strides;

            for (auto i = begin; i != end; ++i)
                shape.push_back(*i);

            for (size_t i = 0; i < shape.size(); ++i) {
                size_t stride = sizeof(Type);
                for (size_t j = i + 1; j < shape.size(); ++j)
                    stride *= shape[j];
                strides.push_back(stride);
            }

            py_array = pybind11::array(pybind11::buffer_info(
                nullptr, sizeof(Type), pybind11::format_descriptor<Type>::value, shape.size(), shape, strides));
            pybind11::buffer_info info = py_array.request();
            Type *ptr = (Type *)info.ptr;

            for (size_t i = 0; i < shape.size(); ++i) {
                strides[i] /= sizeof(Type);
            }
            this->assign(begin, end, strides.begin(), ptr, FirstMajorOrder);
        }

    };





}

namespace tools{

    template<class ARRAY>
    struct BlockStorageSelector;

    template<class T, size_t DIM>
    struct BlockStorageSelector<marray::PyView<T, DIM> >
    {
       typedef BlockView<T> type;
    };
}

}

namespace pybind11
{

    namespace detail
    {

        template <typename Type, size_t DIM> 
        struct pymarray_caster {
            typedef typename nifty::marray::PyView<Type, DIM> ViewType;
            typedef type_caster<typename intrinsic_type<Type>::type> value_conv;

            typedef typename pybind11::array_t<Type> pyarray_type;
            typedef type_caster<pyarray_type> pyarray_conv;

            bool load(handle src, bool convert)
            {
                // convert numpy array to py::array_t
                pyarray_conv conv;
                if (!conv.load(src, convert)){
                    return false;
                }
                auto pyarray = (pyarray_type)conv;

                // convert py::array_t to nifty::marray::PyView
                auto info = pyarray.request();
                if(DIM != 0 && DIM != info.shape.size()){
                    std::cout<<"not matching\n";
                    return false;
                }
                Type *ptr = (Type *)info.ptr;

                ViewType result(pyarray, ptr, info.shape.begin(), info.shape.end());
                value = result;
                return true;
            }

            static handle cast(ViewType src, return_value_policy policy, handle parent)
            {
                pyarray_conv conv;
                return conv.cast(src.py_array, policy, parent);
            }

            PYBIND11_TYPE_CASTER(ViewType, _("array<") + value_conv::name() + _(">"));
        };

        template <typename Type, size_t DIM> 
        struct type_caster<nifty::marray::PyView<Type, DIM> > 
            : pymarray_caster<Type,DIM> {
        };

        //template <typename Type, size_t DIM> 
        //struct marray_caster {
        //    static_assert(std::is_same<Type, void>::value,
        //                  "Please use nifty::marray::PyView instead of nifty::marray::View for arguments and return values.");
        //};
        //template <typename Type> 
        //struct type_caster<andres::View<Type> > 
        //: marray_caster<Type> {
        //};
    }
}














/*

namespace nifty{

    template<class T>
    std::vector<size_t> toSizeT(std::initializer_list<T> l){
        return std::vector<size_t>(l.begin(),l.end());
    }



    template<class T>
    class NumpyArray : public marray::View<T>{
    public:

        template<class SHAPE_T,class STRIDE_T>
        NumpyArray(
            std::initializer_list<SHAPE_T> shape,
            std::initializer_list<STRIDE_T> strides
        )
        : array_(){


            auto svec = toSizeT(strides);
            for(auto i=0; i<svec.size(); ++i){
                svec[i] *= sizeof(T);
            }

            array_  = py::array(
                py::buffer_info(NULL, sizeof(T),
                py::format_descriptor<T>::value,
                shape.size(), toSizeT(shape),svec)
            );

            py::buffer_info info = array_.request();
            T * ptr = static_cast<T*>(info.ptr);

            this->assign(shape.begin(),shape.end(),strides.begin(),ptr,
                marray::FirstMajorOrder
            );
        }


        NumpyArray(py::array_t<T> & array)
        :   array_(array) 
        {
            py::buffer_info info = array.request();
            T * ptr = static_cast<T*>(info.ptr);
            const auto & shape = info.shape;
            auto  strides=  info.strides;
            for(auto & s : strides)
                s/=sizeof(T);
            this->assign(shape.begin(),shape.end(),strides.begin(),ptr,
                marray::FirstMajorOrder
            );
        }
        

        py::array_t<T>  pyArray(){
            return array_;
        }        

    private:
        py::array_t<T> array_;
    };

} // namespace nifty
*/

#endif  // NIFTY_PYTHON_CONVERTER_HXX
