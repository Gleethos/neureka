package neureka.core.module.opti.factory.material;

import neureka.core.T;
import neureka.core.module.opti.TOptimizer;

public abstract class Template implements TOptimizer {

    @Override
    public void optimize(T t){
        double[] value = t.value();//TODO: Add multi-threading!
        t.foreach((i)->this.foreach(i, value));
    }

    protected abstract void foreach(int i, double[] value);



}
