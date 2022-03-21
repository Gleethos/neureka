package neureka.calculus;

import neureka.backend.standard.operations.function.Tanh;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Functions {

    private final Function _dimTrim;
    private final Function _idy;
    private final Function _conv;
    private final Function _plus;
    private final Function _plusAssign;
    private final Function _minus;
    private final Function _minusAssign;
    private final Function _div;
    private final Function _divAssign;
    private final Function _pow;
    private final Function _powAssign;
    private final Function _mul;
    private final Function _mulAssign;
    private final Function _add;
    private final Function _addAssign;
    private final Function _mod;
    private final Function _modAssign;
    private final Function _neg;

    private final Function _matMul;
    private final Function _transpose2D;
    private final Function _random;
    private final Function _tanh;
    private final Function _fastTanh;
    private final Function _softsign;
    private final Function _sigmoid;
    private final Function _gaus;
    private final Function _fastGaus;
    private final Function _ln;
    private final Function _quad;
    private final Function _relu;
    private final Function _abs;
    private final Function _sin;
    private final Function _cos;
    private final Function _softplus;
    private final Function _silu; // Also known as swish!
    private final Function _gelu;

    public Functions( boolean doingAD ) {
        _dimTrim = Function.of( "dimtrim(I[ 0 ])",             doingAD );
        _idy = Function.of( "I[ 0 ]<-I[ 1 ]",                  doingAD );
        _conv = Function.of( "I[ 0 ]xI[ 1 ]",                  doingAD );
        _plus = Function.of( "(I[ 0 ]+I[ 1 ])",                doingAD );
        _plusAssign = Function.of( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",  doingAD );
        _minus = Function.of( "(I[ 0 ]-I[ 1 ])",               doingAD );
        _minusAssign = Function.of( "I[ 0 ]<-(I[ 0 ]-I[ 1 ])", doingAD );
        _div = Function.of( "(I[ 0 ]/I[ 1 ])",                 doingAD );
        _divAssign = Function.of( "I[ 0 ]<-(I[ 0 ]/I[ 1 ])",   doingAD );
        _pow = Function.of( "(I[ 0 ]^I[ 1 ])",                 doingAD );
        _powAssign = Function.of( "I[ 0 ]<-(I[ 0 ]^I[ 1 ])",   doingAD );
        _mul = Function.of( "I[ 0 ]*I[ 1 ]",                   doingAD );
        _mulAssign = Function.of( "I[ 0 ]<-(I[ 0 ]*I[ 1 ])",   doingAD );
        _add = Function.of( "I[ 0 ]+I[ 1 ]",                   doingAD );
        _addAssign = Function.of( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",   doingAD );
        _mod = Function.of( "(I[ 0 ]%I[ 1 ])",                 doingAD );
        _modAssign = Function.of( "I[ 0 ]<-(I[ 0 ]%I[ 1 ])",   doingAD );
        _neg = Function.of( "(-1*I[ 0 ])",                     doingAD );
        _matMul = Function.of("I[0] @ I[1]",                   doingAD );
        _transpose2D = Function.of("[1, 0]:(I[0])",            doingAD );
        _random = Function.of("random(I[0])",                  doingAD );
        _tanh = Function.of( "tanh(I[0])",                     doingAD );
        _fastTanh = Function.of( "fast_tanh(I[0])",            doingAD );
        _softsign = Function.of( "softsign(I[0])",            doingAD );
        _sigmoid = Function.of( "sig(I[0])",                   doingAD );
        _gaus = Function.of("gaus(I[0])",                      doingAD );
        _fastGaus = Function.of("fast_gaus(I[0])",             doingAD );
        _ln = Function.of("ln(I[0])",                          doingAD );
        _quad = Function.of("quad(I[0])",                      doingAD );
        _relu = Function.of("relu(I[0])",                      doingAD );
        _abs  = Function.of("abs(I[0])",                       doingAD );
        _sin  = Function.of("sin(I[0])",                       doingAD );
        _cos  = Function.of("cos(I[0])",                       doingAD );
        _softplus  = Function.of("softplus(I[0])",             doingAD );
        _silu  = Function.of("silu(I[0])",                     doingAD );
        _gelu  = Function.of("gelu(I[0])",                              doingAD );
    }

    public final Function getDimTrim() { return _dimTrim; }

    public final Function dimTrim() { return _dimTrim; }

    public final Function getIdy() { return _idy; }

    public final Function idy() { return _idy; }

    public final Function getConv() { return _conv; }

    public final Function conv() { return _conv; }

    public final Function getPlus() { return _plus; }

    public final Function plus() { return _plus; }

    public final Function getPlusAssign() { return _plusAssign; }

    public final Function plusAssign() { return _plusAssign; }

    public final Function getMinus() { return _minus; }

    public final Function minus() { return _minus; }

    public final Function getMinusAssign() { return _minusAssign; }

    public final Function minusAssign() { return _minusAssign; }

    public final Function getDiv() { return _div; }

    public final Function div() { return _div; }

    public final Function getDivAssign() { return _divAssign; }

    public final Function divAssign() { return _divAssign; }

    public final Function getPow() { return _pow; }

    public final Function pow() { return _pow; }

    public final Function getPowAssign() { return _powAssign; }

    public final Function powAssign() { return _powAssign; }

    public final Function getMul() { return _mul; }

    public final Function mul() { return _mul; }

    public final Function getMulAssign() { return _mulAssign; }

    public final Function mulAssign() { return _mulAssign; }

    public final Function getAdd() { return _add; }

    public final Function add() { return _add; }

    public final Function getAddAssign() { return _addAssign; }

    public final Function addAssign() { return _addAssign; }

    public final Function getMod() { return _mod; }

    public final Function mod() { return _mod; }

    public final Function getModAssign() { return _modAssign; }

    public final Function modAssign() { return _modAssign; }

    public final Function getNeg() { return _neg; }

    public final Function neg() { return _neg; }

    public final Function getMatMul() { return _matMul; }

    public final Function matMul() { return _matMul; }

    public final Function getTranspose2D() { return _transpose2D; }

    public final Function transpose2D() { return _transpose2D; }

    public final Function getRandom() { return _random; }

    public final Function random() { return _random; }

    /**
     * @return A tanh {@link Function} based on: {@code 2 / ( 1 + Math.exp( -x * 2 ) ) - 1}.
     */
    public final Function getTanh() { return _tanh; }

    /**
     * @return A tanh {@link Function} based on: {@code 2 / ( 1 + Math.exp( -x * 2 ) ) - 1}.
     */
    public final Function tanh() { return _tanh; }

    /**
     * @return A fast quasi tanh {@link Function} based on: {@code x * FastFun.invSqrt( 1 + x * x )}.
     */
    public final Function getFastTanh() { return _fastTanh; }

    /**
     * @return A fast quasi tanh {@link Function} based on: {@code x * FastFun.invSqrt( 1 + x * x )}.
     */
    public final Function fastTanh() { return _fastTanh; }

    /**
     *  The softsign function, defined as {@code x / ( 1 + Math.abs( x ) )},
     *  is a computationally cheap 0 centered activation function
     *  which rescales the inputs between -1 and 1, very much like the {@link Tanh} function.
     *  The softsign function converges polynomially and is computationally cheaper than the
     *  tanh function which converges exponentially.
     *
     * @return A very fast quasi tanh {@link Function} based on: {@code x / ( 1 + Math.abs( x ) )}.
     */
    public final Function getSoftsign() { return _softsign; }

    /**
     *  The softsign function, defined as {@code x / ( 1 + Math.abs( x ) )},
     *  is a computationally cheap 0 centered activation function
     *  which rescales the inputs between -1 and 1, very much like the {@link Tanh} function.
     *  The softsign function converges polynomially and is computationally cheaper than the
     *  tanh function which converges exponentially.
     *
     * @return A very fast quasi tanh {@link Function} based on: {@code x / ( 1 + Math.abs( x ) )}.
     */
    public final Function softsign() { return _softsign; }

    /**
     * @return A sigmoid {@link Function} based on: {@code 1 / ( 1 + Math.exp( -x ) )}.
     */
    public final Function getSigmoid() { return _sigmoid; }

    /**
     * @return A sigmoid {@link Function} based on: {@code 1 / ( 1 + Math.exp( -x ) )}.
     */
    public final Function sigmoid() { return _sigmoid; }

    /**
     * @return A gaussian {@link Function} based on: {@code Math.exp( -( x * x ) )}.
     */
    public final Function getGaus() { return _gaus; }

    /**
     * @return A gaussian {@link Function} based on: {@code Math.exp( -( x * x ) )}.
     */
    public final Function gaus() { return _gaus; }

    /**
     * @return A quasi gaussian {@link Function} based on: {@code 1 / ( 1 + x * x )}.
     */
    public final Function getFastGaus () { return _fastGaus; }

    /**
     * @return A quasi gaussian {@link Function} based on: {@code 1 / ( 1 + x * x )}.
     */
    public final Function fastGaus () { return _fastGaus; }

    /**
     * @return A natural log {@link Function} based on: {@code Math.log( x )}.
     */
    public final Function getLn() { return _ln; }

    /**
     * @return A natural log {@link Function} based on: {@code Math.log( x )}.
     */
    public final Function ln() { return _ln; }

    /**
     * @return A quadratic {@link Function} based on: {@code x * x}.
     */
    public final Function getQuad() { return _quad; }

    /**
     * @return A quadratic {@link Function} based on: {@code x * x}.
     */
    public final Function quad() { return _quad; }

    /**
     * @return A rectified linear unit {@link Function} based on: {@code ( x >= 0 ? x : x * .01 )}.
     */
    public final Function getRelu() { return _relu; }

    /**
     * @return A rectified linear unit {@link Function} based on: {@code ( x >= 0 ? x : x * .01 )}.
     */
    public final Function relu() { return _relu; }

    /**
     * @return An absolute {@link Function} based on: {@code Math.abs(x)}.
     */
    public final Function getAbs() { return _abs; }

    /**
     * @return An absolute {@link Function} based on: {@code Math.abs(x)}.
     */
    public final Function abs() { return _abs; }

    /**
     * @return A sine {@link Function} based on: {@code Math.sin(x)}.
     */
    public final Function getSin() { return _sin; }

    /**
     * @return A sine {@link Function} based on: {@code Math.sin(x)}.
     */
    public final Function sin() { return _sin; }

    /**
     * @return A cosine {@link Function} based on: {@code Math.cos(x)}.
     */
    public final Function getCos() { return _cos; }

    /**
     * @return A cosine {@link Function} based on: {@code Math.cos(x)}.
     */
    public final Function cos() { return _cos; }

    /**
     *  SoftPlus is a smooth approximation to the ReLU function and can be
     *  used to constrain the output of a machine to always be positive.
     *
     * @return A softplus {@link Function} based on: {@code Math.log( 1 + Math.exp( x ) )}.
     */
    public final Function getSoftplus() { return _softplus; }

    /**
     *  SoftPlus is a smooth approximation to the ReLU function and can be
     *  used to constrain the output of a machine to always be positive.
     *
     * @return A softplus {@link Function} based on: {@code Math.log( 1 + Math.exp( x ) )}.
     */
    public final Function softplus() { return _softplus; }

    /**
     *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
     *  It is a smooth, non-monotonic function that consistently matches
     *  or outperforms ReLU on deep networks,
     *  it is unbounded above and bounded below.
     *
     * @return A SiLu {@link Function} (also known as swish) based on: {@code x / ( 1 + Math.exp( -x ) )}.
     */
    public final Function getSilu() { return _silu; }

    /**
     *  The SiLu activation function, also known as the swish function, is defined as {@code x * sigmoid(x)}.
     *  It is a smooth, non-monotonic function that consistently matches
     *  or outperforms ReLU on deep networks,
     *  it is unbounded above and bounded below.
     *
     * @return A SiLu {@link Function} (also known as swish) based on: {@code x / ( 1 + Math.exp( -x ) )}.
     */
    public final Function silu() { return _silu; }

    /**
     * @return A GeLU {@link Function} based on: {@code x / ( 1 + Math.exp( -x * 1.702 ) )}.
     */
    public final Function getGelu() { return _gelu; }

    /**
     * @return A GeLU {@link Function} based on: {@code x / ( 1 + Math.exp( -x * 1.702 ) )}.
     */
    public final Function gelu() { return _gelu; }



    @Override
    public final String toString() {
        String state =
                Arrays.stream(this.getClass().getDeclaredFields())
                      .map( field -> {
                          try {
                              return field.getName()+"="+field.get(this).toString()+"";
                          } catch (IllegalAccessException e) {
                              return field.getName()+"=?";
                          }
                      })
                      .collect(Collectors.joining(","));

        return this.getClass().getSimpleName()+"["+state+"]";
    }

}
