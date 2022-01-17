package neureka.calculus;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Functions {

    private final Function dimTrim;
    private final Function idy;
    private final Function conv;
    private final Function plus;
    private final Function plusAssign;
    private final Function minus;
    private final Function minusAssign;
    private final Function div;
    private final Function divAssign;
    private final Function pow;
    private final Function powAssign;
    private final Function mul;
    private final Function mulAssign;
    private final Function add;
    private final Function addAssign;
    private final Function mod;
    private final Function modAssign;
    private final Function neg;

    private final Function matMul;

    public Functions( boolean doingAD ) {
        dimTrim = Function.of( "dimtrim(I[ 0 ])",             doingAD );
        idy = Function.of( "I[ 0 ]<-I[ 1 ]",                  doingAD );
        conv = Function.of( "I[ 0 ]xI[ 1 ]",                  doingAD );
        plus = Function.of( "(I[ 0 ]+I[ 1 ])",                doingAD );
        plusAssign = Function.of( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",  doingAD );
        minus = Function.of( "(I[ 0 ]-I[ 1 ])",               doingAD );
        minusAssign = Function.of( "I[ 0 ]<-(I[ 0 ]-I[ 1 ])", doingAD );
        div = Function.of( "(I[ 0 ]/I[ 1 ])",                 doingAD );
        divAssign = Function.of( "I[ 0 ]<-(I[ 0 ]/I[ 1 ])",   doingAD );
        pow = Function.of( "(I[ 0 ]^I[ 1 ])",                 doingAD );
        powAssign = Function.of( "I[ 0 ]<-(I[ 0 ]^I[ 1 ])",   doingAD );
        mul = Function.of( "I[ 0 ]*I[ 1 ]",                   doingAD );
        mulAssign = Function.of( "I[ 0 ]<-(I[ 0 ]*I[ 1 ])",   doingAD );
        add = Function.of( "I[ 0 ]+I[ 1 ]",                   doingAD );
        addAssign = Function.of( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",   doingAD );
        mod = Function.of( "(I[ 0 ]%I[ 1 ])",                 doingAD );
        modAssign = Function.of( "I[ 0 ]<-(I[ 0 ]%I[ 1 ])",   doingAD );
        neg = Function.of( "(-1*I[ 0 ])",                     doingAD );
        matMul = Function.of("I[0] @ I[1]",                   doingAD );
    }


    public Function getDimTrim() { return this.dimTrim; }

    public Function getIdy() { return this.idy; }

    public Function getConv() { return this.conv; }

    public Function getPlus() { return this.plus; }

    public Function getPlusAssign() { return this.plusAssign; }

    public Function getMinus() { return this.minus; }

    public Function getMinusAssign() { return this.minusAssign; }

    public Function getDiv() { return this.div; }

    public Function getDivAssign() { return this.divAssign; }

    public Function getPow() { return this.pow; }

    public Function getPowAssign() { return this.powAssign; }

    public Function getMul() { return this.mul; }

    public Function getMulAssign() { return this.mulAssign; }

    public Function getAdd() { return this.add; }

    public Function getAddAssign() { return this.addAssign; }

    public Function getMod() { return this.mod; }

    public Function getModAssign() { return this.modAssign; }

    public Function getNeg() { return this.neg; }

    public Function dimTrim() {
        return this.dimTrim;
    }

    public Function idy() {
        return this.idy;
    }

    public Function conv() {
        return this.conv;
    }

    public Function plus() {
        return this.plus;
    }

    public Function plusAssign() {
        return this.plusAssign;
    }

    public Function minus() {
        return this.minus;
    }

    public Function minusAssign() {
        return this.minusAssign;
    }

    public Function div() {
        return this.div;
    }

    public Function divAssign() {
        return this.divAssign;
    }

    public Function pow() {
        return this.pow;
    }

    public Function powAssign() {
        return this.powAssign;
    }

    public Function mul() {
        return this.mul;
    }

    public Function mulAssign() {
        return this.mulAssign;
    }

    public Function add() {
        return this.add;
    }

    public Function addAssign() {
        return this.addAssign;
    }

    public Function mod() {
        return this.mod;
    }

    public Function modAssign() {
        return this.modAssign;
    }

    public Function neg() {
        return this.neg;
    }

    public Function matMul() { return this.matMul; }

    @Override
    public String toString() {
        String state = Arrays.stream(this.getClass().getDeclaredFields())
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
