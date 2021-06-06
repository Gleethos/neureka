package neureka.calculus;

import lombok.Getter;
import lombok.experimental.Accessors;

@Accessors( fluent = true )
public class Functions {

    @Getter Function dimTrim;
    @Getter Function idy;
    @Getter Function conv;
    @Getter Function plus;
    @Getter Function plusAssign;
    @Getter Function minus;
    @Getter Function minusAssign;
    @Getter Function div;
    @Getter Function divAssign;
    @Getter Function pow;
    @Getter Function powAssign;
    @Getter Function mul;
    @Getter Function mulAssign;
    @Getter Function add;
    @Getter Function addAssign;
    @Getter Function mod;
    @Getter Function modAssign;
    @Getter Function neg;

    public Functions( boolean doingAD ) {
        dimTrim = Function.create( "dimtrim(I[ 0 ])",             doingAD );
        idy = Function.create( "I[ 0 ]<-I[ 1 ]",                  doingAD );
        conv = Function.create( "I[ 0 ]xI[ 1 ]",                  doingAD );
        plus = Function.create( "(I[ 0 ]+I[ 1 ])",                doingAD );
        plusAssign = Function.create( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",  doingAD );
        minus = Function.create( "(I[ 0 ]-I[ 1 ])",               doingAD );
        minusAssign = Function.create( "I[ 0 ]<-(I[ 0 ]-I[ 1 ])", doingAD );
        div = Function.create( "(I[ 0 ]/I[ 1 ])",                 doingAD );
        divAssign = Function.create( "I[ 0 ]<-(I[ 0 ]/I[ 1 ])",   doingAD );
        pow = Function.create( "(I[ 0 ]^I[ 1 ])",                 doingAD );
        powAssign = Function.create( "I[ 0 ]<-(I[ 0 ]^I[ 1 ])",   doingAD );
        mul = Function.create( "I[ 0 ]*I[ 1 ]",                   doingAD );
        mulAssign = Function.create( "I[ 0 ]<-(I[ 0 ]*I[ 1 ])",   doingAD );
        add = Function.create( "I[ 0 ]+I[ 1 ]",                   doingAD );
        addAssign = Function.create( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])",   doingAD );
        mod = Function.create( "(I[ 0 ]%I[ 1 ])",                 doingAD );
        modAssign = Function.create( "I[ 0 ]<-(I[ 0 ]%I[ 1 ])",   doingAD );
        neg = Function.create( "(-1*I[ 0 ])",                     doingAD );
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
}
