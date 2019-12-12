package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.function.Function;
import neureka.function.factory.assembly.FunctionBuilder;

public class ADAM {

    //VARIABLES...
    private Tsr a;
    private Tsr b1;
    private Tsr b2;
    private Tsr e;
    Tsr m = null;
    Tsr v = null;
    Tsr w = null;

    ADAM(Tsr target){
        w = target;
        int[] shape = target.shape();
        m = new Tsr(shape, 0);
        v = new Tsr(shape, 0);
        a = new Tsr(shape, 0.001);
        b1 = new Tsr(shape, 0.9);
        b2 = new Tsr(shape, 0.999);
        e = new Tsr(shape, 0.000000001);
    }

    private void _optimize(){
        Function inject = FunctionBuilder.build("I[0]<<xIg[1]", false);
        Tsr g = new Tsr(w.shape());
        inject.activate(new Tsr[]{g, w});

        m = new Tsr(b1, "*", m, "+(1-", b1, ")*", g);
        v = new Tsr(b2, "*", v, "+(1-", b2, ")*", g);

        Tsr mh = null;
        Tsr vh = null;

        mh = new Tsr(m, "/(1-", b1, ")");
        vh = new Tsr(v, "/(1-", b2, ")");

        Tsr newW = new Tsr(w,"-",a,"*(",mh,"/(",vh,"^-2+",e,"))");
        Function f = FunctionBuilder.build("I[0]<<xI[1]", false);
        Function r = FunctionBuilder.build("Ig[0]<<x0", false);
        f.activate(new Tsr[]{w, newW});
        r.activate(new Tsr[]{w});
    }

}
