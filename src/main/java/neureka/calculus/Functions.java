package neureka.calculus;

import lombok.Getter;
import lombok.experimental.Accessors;

@Accessors( fluent = true )
public class Functions {

    @Getter Function DIMTRIM     ;
    @Getter Function IDY         ;
    @Getter Function X           ;
    @Getter Function PLUS        ;
    @Getter Function PLUS_ASSIGN ;
    @Getter Function MINUS       ;
    @Getter Function MINUS_ASSIGN;
    @Getter Function DIV         ;
    @Getter Function DIV_ASSIGN  ;
    @Getter Function POW         ;
    @Getter Function POW_ASSIGN  ;
    @Getter Function MUL         ;
    @Getter Function MUL_ASSIGN  ;
    @Getter Function ADD         ;
    @Getter Function ADD_ASSIGN  ;
    @Getter Function MOD         ;
    @Getter Function MOD_ASSIGN  ;
    @Getter Function NEG         ;

    public Functions( boolean doingAD ) {
        DIMTRIM       = Function.create( "dimtrim(I[ 0 ])",         doingAD );
        IDY           = Function.create( "I[ 0 ]<-I[ 1 ]",          doingAD );
        X             = Function.create( "I[ 0 ]xI[ 1 ]",           doingAD );
        PLUS          = Function.create( "(I[ 0 ]+I[ 1 ])",         doingAD );
        PLUS_ASSIGN   = Function.create( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", doingAD );
        MINUS         = Function.create( "(I[ 0 ]-I[ 1 ])",         doingAD );
        MINUS_ASSIGN  = Function.create( "I[ 0 ]<-(I[ 0 ]-I[ 1 ])", doingAD );
        DIV           = Function.create( "(I[ 0 ]/I[ 1 ])",         doingAD );
        DIV_ASSIGN    = Function.create( "I[ 0 ]<-(I[ 0 ]/I[ 1 ])", doingAD );
        POW           = Function.create( "(I[ 0 ]^I[ 1 ])",         doingAD );
        POW_ASSIGN    = Function.create( "I[ 0 ]<-(I[ 0 ]^I[ 1 ])", doingAD );
        MUL           = Function.create( "I[ 0 ]*I[ 1 ]",           doingAD );
        MUL_ASSIGN    = Function.create( "I[ 0 ]<-(I[ 0 ]*I[ 1 ])", doingAD );
        ADD           = Function.create( "I[ 0 ]+I[ 1 ]",           doingAD );
        ADD_ASSIGN    = Function.create( "I[ 0 ]<-(I[ 0 ]+I[ 1 ])", doingAD );
        MOD           = Function.create( "(I[ 0 ]%I[ 1 ])",         doingAD );
        MOD_ASSIGN    = Function.create( "I[ 0 ]<-(I[ 0 ]%I[ 1 ])", doingAD );
        NEG           = Function.create( "(-1*I[ 0 ])",             doingAD );
    }


}
