package neureka.optimization.implementations;

import neureka.Tsr;
import neureka.optimization.Optimizer;

public abstract class Template implements Optimizer {

    @Override
    public void optimize(Tsr t){
        //double[] value = t.value64();//TODO: Add multi-threading!
        //t.foreach((i)->this.foreach(i, value));
    }



}
