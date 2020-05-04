package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.calculus.Function;
import neureka.optimization.Optimizer;

public class ADAM implements Optimizer {

    //VARIABLES...
    private Tsr a;
    private Tsr b1;
    private Tsr b2;
    private Tsr e;
    Tsr m = null;
    Tsr v = null;
    //Tsr w = null;

    ADAM(Tsr target){
        int[] shape = target.shape();
        m = new Tsr(shape, 0);
        v = new Tsr(shape, 0);
        a = new Tsr(shape, 0.01); // Step size!
        b1 = new Tsr(shape, 0.9);
        b2 = new Tsr(shape, 0.999);
        e = new Tsr(shape, 1e-7);
    }

    private void _optimize(Tsr w){
        Tsr g = (Tsr)w.find(Tsr.class);
        m = new Tsr(b1, "*", m, " + ( 1-", b1, ") *", g);
        v = new Tsr(b2, "*", v, " + ( 1-", b2, ") * (", g,"^2 )");
        Tsr mh = new Tsr(m, "/(1-", b1, ")");
        Tsr vh = new Tsr(v, "/(1-", b2, ")");
        Tsr newg = new Tsr("-",a,"*",mh,"/(",vh,"^0.5+",e,")");
        Function.Detached.IDY.activate(new Tsr[]{g, newg});
    }

    @Override
    public void optimize(Tsr t) {
        _optimize(t);
    }

}
