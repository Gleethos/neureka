package neureka.calculus;

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
    }

    public Function getDimTrim() { return _dimTrim; }

    public Function getIdy() { return _idy; }

    public Function getConv() { return _conv; }

    public Function getPlus() { return _plus; }

    public Function getPlusAssign() { return _plusAssign; }

    public Function getMinus() { return _minus; }

    public Function getMinusAssign() { return _minusAssign; }

    public Function getDiv() { return _div; }

    public Function getDivAssign() { return _divAssign; }

    public Function getPow() { return _pow; }

    public Function getPowAssign() { return _powAssign; }

    public Function getMul() { return _mul; }

    public Function getMulAssign() { return _mulAssign; }

    public Function getAdd() { return _add; }

    public Function getAddAssign() { return _addAssign; }

    public Function getMod() { return _mod; }

    public Function getModAssign() { return _modAssign; }

    public Function getNeg() { return _neg; }

    public Function getMatMul() { return _matMul; }

    public Function getTranspose2D() { return _transpose2D; }

    public Function dimTrim() {
        return _dimTrim;
    }

    public Function idy() {
        return _idy;
    }

    public Function conv() {
        return _conv;
    }

    public Function plus() {
        return _plus;
    }

    public Function plusAssign() {
        return _plusAssign;
    }

    public Function minus() {
        return _minus;
    }

    public Function minusAssign() {
        return _minusAssign;
    }

    public Function div() {
        return _div;
    }

    public Function divAssign() {
        return _divAssign;
    }

    public Function pow() {
        return _pow;
    }

    public Function powAssign() {
        return _powAssign;
    }

    public Function mul() {
        return _mul;
    }

    public Function mulAssign() {
        return _mulAssign;
    }

    public Function add() {
        return _add;
    }

    public Function addAssign() {
        return _addAssign;
    }

    public Function mod() {
        return _mod;
    }

    public Function modAssign() {
        return _modAssign;
    }

    public Function neg() {
        return _neg;
    }

    public Function matMul() { return _matMul; }

    public Function transpose2D() { return _transpose2D; }

    @Override
    public String toString() {
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
