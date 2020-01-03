// MIT License
//
// Copyright (c) 2019 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef MIYUKI_VECTORIZE_VECTORIZE_HPP
#define MIYUKI_VECTORIZE_VECTORIZE_HPP

#include <array>
#include <type_traits>
#include <functional>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cstring>

namespace miyuki::vectorize {


    template<class T, int N>
    class Array;

    constexpr int align4(int N) {
        return (3 + N) & (-4);
    }


    namespace detail {
        template<class T, int N>
        struct to_array {
            using type = Array<T, N>;

        };
    }


    template<class F, int N, class... Args>
    auto
    apply(F &&f,
          const Array<Args, N> &... args) -> typename detail::to_array<std::invoke_result_t<F, Args...>, N>::type {
        typename detail::to_array<std::invoke_result_t<F, Args...>, N>::type ret;
        for (auto i = 0; i < N; i++) {
            ret[i] = f(args[i]...);
        }
        return ret;
    }


#define MYK_VEC_ARR_GEN_OP_(op) friend self_type operator op (const self_type & lhs,const self_type & rhs ) {\
                                        return apply([](const value_type &a, const value_type &b)->value_type{\
                                            return value_type(a op b);\
                                        }, lhs, rhs); \
                                }
#define MYK_VEC_ARR_GEN_OP(op) MYK_VEC_ARR_GEN_OP_(op)
#define MYK_VEC_ARR_GEN_ASSIGN_OP(op) self_type& operator op##= (const self_type & rhs){*this = *this op rhs;return *this;}

#define MYK_VEC_GEN_BASIC_ASSIGN_OPS()  \
    MYK_VEC_ARR_GEN_ASSIGN_OP(+) \
    MYK_VEC_ARR_GEN_ASSIGN_OP(-)\
    MYK_VEC_ARR_GEN_ASSIGN_OP(*)\
    MYK_VEC_ARR_GEN_ASSIGN_OP(/)

#define MYK_VEC_GEN_BASIC_OPS() \
    MYK_VEC_ARR_GEN_OP_(+) \
    MYK_VEC_ARR_GEN_OP_(-)\
    MYK_VEC_ARR_GEN_OP_(*)\
    MYK_VEC_ARR_GEN_OP_(/)\
    MYK_VEC_ARR_GEN_OP_(==)\
    MYK_VEC_ARR_GEN_OP_(!=)\
    MYK_VEC_ARR_GEN_OP_(<)\
    MYK_VEC_ARR_GEN_OP_(<=)\
    MYK_VEC_ARR_GEN_OP_(>)\
    MYK_VEC_ARR_GEN_OP_(>=)

    template<class T, int N>
    class Array : public std::array<T, N> {
        using value_type = T;
        using self_type = Array<T, N>;
    public:
        Array() = default;

        Array(std::initializer_list<T> list) {
            auto it = list.begin();
            for (int i = 0; i < 4; i++) {
                (*this)[i] = *it;
                it++;
            }
        }

        explicit Array(const T &value) {
            for (auto &i: *this) {
                i = value;
            }
        }

        MYK_VEC_GEN_BASIC_OPS()

        MYK_VEC_GEN_BASIC_ASSIGN_OPS()
    };

#define MYK_VEC_GEN_ACCESS(Name, Index) \
    auto Name ()const {static_assert(Index < N, "component "#Name "is not available");return this->s[Index];} \
    auto& Name() {static_assert(Index < N, "component "#Name "is not available");return this->s[Index];}

    template<int N>
    class Array<float, N> : public std::array<float, N> {
        using value_type = float;
        using self_type = Array;
    public:
        MYK_VEC_GEN_BASIC_OPS()

        MYK_VEC_GEN_BASIC_ASSIGN_OPS()
    };

    class Float4Base {
        using self_type = Float4Base;
        using value_type = float;
    public:
        union {
            float s[4];
            __m128 m;
        };

        float &operator[](int i) {
            return s[i];
        }

        const float &operator[](int i) const {
            return s[i];
        }

        Float4Base() = default;

        explicit Float4Base(const float &v) : m(_mm_broadcast_ss(&v)) {}

        Float4Base(std::initializer_list<float> list) {
            auto it = list.begin();
            for (int i = 0; i < 4; i++) {
                (*this)[i] = *it;
                it++;
            }
        }

        Float4Base(__m128 m) {
            auto n = m;
            memcpy(this, &n, sizeof(__m128));
        }

        operator __m128() const {
            __m128 m;
            memcpy(&m, this, sizeof(__m128));
            return m;
        };

        friend Float4Base operator+(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_add_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator-(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_sub_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator*(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_mul_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator/(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_div_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator<(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_cmplt_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator<=(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_cmple_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator>(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_cmpgt_ps((__m128) lhs, (__m128) rhs));
        }

        friend Float4Base operator>=(const Float4Base &lhs, const Float4Base &rhs) {
            return Float4Base(_mm_cmpge_ps((__m128) lhs, (__m128) rhs));
        }


        MYK_VEC_GEN_BASIC_ASSIGN_OPS()
    };


    class Float8Base {
        using self_type = Float8Base;
        using value_type = float;
    public:
        union {
            float s[8];
            __m256 m;
        };

        explicit Float8Base(const float &v) : m(_mm256_broadcast_ss(&v)) {}

        Float8Base(__m256 m) {
            auto n = m;
            memcpy(&this->m, &n, sizeof(__m256));
        }

        operator __m256() const {
            __m256 m;
            memcpy(&m, &this->m, sizeof(__m256));
            return m;
        };

        friend Float8Base operator+(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_add_ps((__m256) lhs, (__m256) rhs));
        }

        friend Float8Base operator-(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_sub_ps((__m256) lhs, (__m256) rhs));
        }

        friend Float8Base operator*(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_mul_ps((__m256) lhs, (__m256) rhs));
        }

        friend Float8Base operator/(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_div_ps((__m256) lhs, (__m256) rhs));
        }

        friend Float8Base operator<(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_cmp_ps((__m256) lhs, (__m256) rhs, _CMP_LT_OQ));
        }

        friend Float8Base operator<=(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_cmp_ps((__m256) lhs, (__m256) rhs, _CMP_LE_OQ));
        }

        friend Float8Base operator>(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_cmp_ps((__m256) lhs, (__m256) rhs, _CMP_GT_OQ));
        }

        friend Float8Base operator>=(const Float8Base &lhs, const Float8Base &rhs) {
            return Float8Base(_mm256_cmp_ps((__m256) lhs, (__m256) rhs, _CMP_GT_OQ));
        }


        MYK_VEC_GEN_BASIC_ASSIGN_OPS()
    };

    static_assert(sizeof(__m128) == sizeof(Float4Base));

    template<>
    class Array<float, 4> : public Float4Base {
        static const int N = 4;
    public:
        using Float4Base::Float4Base;

        MYK_VEC_GEN_ACCESS(x, 0)

        MYK_VEC_GEN_ACCESS(y, 1)

        MYK_VEC_GEN_ACCESS(z, 2)

        MYK_VEC_GEN_ACCESS(w, 3)

    };

    template<>
    class Array<float, 3> : public Float4Base {
        static const int N = 3;
    public:
        using Float4Base::Float4Base;

        MYK_VEC_GEN_ACCESS(x, 0)

        MYK_VEC_GEN_ACCESS(y, 1)

        MYK_VEC_GEN_ACCESS(z, 2)
    };

    template<class T, size_t N>
    T dot(const Array<T, N> &a, const Array<T, N> &b) {
        auto x = a[0] * b[0];
        for (auto i = 1; i < N; i++) {
            x += a[i] * b[i];
        }
        return x;
    }

    template<class T, size_t N>
    Array<T, N> normalize(const Array<T, N> &v) {
        return v / Array<T, N>(dot(v, v));
    }

    template<class T>
    T cross(const Array<T, 3> &v1, const Array<T, 3> &v2) {
        return {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
    }


    template<class T>
    class Matrix4 {
        Array<Array<T, 4>, 4> _rows;
        static_assert(sizeof(_rows) == sizeof(T) * 16);
        using Vec3 = Array<T, 3>;
    public:
        static Matrix4 zero() {
            Matrix4 matrix4;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    matrix4[i][j] = T(0.0f);
                }
            }
            return matrix4;
        }

        static Matrix4 identity() {
            Matrix4 matrix4;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    matrix4[i][j] = i == j ? T(1.0f) : T(0.0f);
                }
            }
            return matrix4;
        }

        static Matrix4 scale(const T & k) {
            Matrix4 matrix4;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    matrix4[i][j] = i == j ? k : T(0.0f);
                }
            }
            return matrix4;
        }
        static Matrix4 rotate(const Vec3 &axis, const T& angle) {
            const auto zero = T(0.0f);
            const auto one = T(1.0f);
            const Float s = sin(angle);
            const Float c = cos(angle);
            const Float oc = one - c;
            T r[4][4] = {
                    {oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, zero},
                    {oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s, zero},
                    {oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c, zero},
                    {zero,zero,zero,one}};
            Matrix4 m;
            std::memcpy(&m, &r, sizeof(T) * 16);
            return m;
        }
        static Matrix4 translate(const Vec3 & v) {
            Matrix4 m = identity();
            m[0][3] = v[0];
            m[1][3] = v[1];
            m[2][3] = v[2];
            return  m;
        }

        Matrix4 inverse() const {
            auto m = reinterpret_cast<const T *>(_rows);
            T inv[16], det;
            int i;


            inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
                     m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

            inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
                     m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

            inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
                     m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

            inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
                      m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

            inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
                     m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

            inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
                     m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

            inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
                     m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

            inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
                      m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

            inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
                     m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

            inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
                     m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

            inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
                      m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

            inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
                      m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

            inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
                     m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

            inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
                     m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

            inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
                      m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

            inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
                      m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

            det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

            if (det == 0) {
                if (suc) {
                    *suc = false;
                }
            }

            det = T(1.0) / det;

            Matrix4 <T> out;
            auto invOut = reinterpret_cast<T *>(out._rows);
            for (i = 0; i < 16; i++)
                invOut[i] = inv[i] * det;
            return out;
        }

        Matrix4 operator*(const Matrix4 &rhs) const {
            Matrix4 m;
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    m._rows[i][j] = dot(_rows[i].rhs.column(j));
                }
            }
            return m;
        }

        Matrix4 &operator*=(const Matrix4 &rhs) {
            auto m = *this * rhs;
            std::memcpy(this, m, sizeof(T* 16));
            return *this;
        }

        Array<T, 4> operator*(const Array<T, 4> &v) const {
            return Array<T, 4>{dot(_rows[0],v),
                               dot(_rows[1],v),
                               dot(_rows[2],v),
                               dot(_rows[3],v)};
        }

        Array<T, 4> column(size_t i) const {
            return Array<T, 4>{_rows[0][i], _rows[1][i], _rows[2][i], _rows[3][i]};
        }
    };
}

namespace miyuki {
#define MYK_VEC_DECL_PRIMITIVE(TY, N) using TY##N = vectorize::Array<TY, N>;
    MYK_VEC_DECL_PRIMITIVE(bool, 2)
    MYK_VEC_DECL_PRIMITIVE(bool, 3)
    MYK_VEC_DECL_PRIMITIVE(bool, 4)
    MYK_VEC_DECL_PRIMITIVE(bool, 8)
    MYK_VEC_DECL_PRIMITIVE(bool, 16)

    MYK_VEC_DECL_PRIMITIVE(float, 2)
    MYK_VEC_DECL_PRIMITIVE(float, 3)
    MYK_VEC_DECL_PRIMITIVE(float, 4)
    MYK_VEC_DECL_PRIMITIVE(float, 8)
    MYK_VEC_DECL_PRIMITIVE(float, 16)

    MYK_VEC_DECL_PRIMITIVE(int, 2)
    MYK_VEC_DECL_PRIMITIVE(int, 3)
    MYK_VEC_DECL_PRIMITIVE(int, 4)
    MYK_VEC_DECL_PRIMITIVE(int, 8)
    MYK_VEC_DECL_PRIMITIVE(int, 16)

}


#define MYK_VEC_STRUCT_BEGIN(Struct) template<int NElem> struct T##Struct { using Base = Struct;
#define MYK_VEC_MEMBER(Name) typename miyuki::vectorize::detail::to_array<decltype(Base::Name),NElem>::type Name;
#define MYK_VEC_METHOD(Ret, Name, ...) Ret Name()
#define MYK_VEC_STRUCT_END };

#endif //MIYUKI_VECTORIZE_VECTORIZE_HPP
