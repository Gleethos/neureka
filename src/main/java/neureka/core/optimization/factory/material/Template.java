package neureka.core.optimization.factory.material;

import neureka.core.Tsr;
import neureka.core.optimization.TOptimizer;

public abstract class Template implements TOptimizer {

    @Override
    public void optimize(Tsr t){
        double[] value = t.value64();//TODO: Add multi-threading!
        t.foreach((i)->this.foreach(i, value));
    }

    protected abstract void foreach(int i, double[] value);



}
