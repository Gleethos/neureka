package neureka.calculus;

public class Functions {

    Function dimTrim;
    Function idy;
    Function conv;
    Function plus;
    Function plusAssign;
    Function minus;
    Function minusAssign;
    Function div;
    Function divAssign;
    Function pow;
    Function powAssign;
    Function mul;
    Function mulAssign;
    Function add;
    Function addAssign;
    Function mod;
    Function modAssign;
    Function neg;

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
}
