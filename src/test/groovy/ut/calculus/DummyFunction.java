package ut.calculus;

import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.calculus.Function;
import neureka.calculus.args.Args;

import java.util.List;
import java.util.function.BiFunction;

public class DummyFunction implements Function {

    private final BiFunction<Object, Object, Tsr<?>> implementation;

    DummyFunction( BiFunction<Object, Object, Tsr<?>> implementation ) {
        this.implementation = implementation;
    }

    @Override
    public Tsr<?> execute(Args arguments, Tsr<?>... tensors) {
        return implementation.apply( arguments, tensors.clone() );
    }

    @Override public boolean isDoingAD() { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public boolean isFlat() { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public Operation getOperation() { return null; }
    @Override public boolean dependsOn(int index) { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public Function getDerivative(int index) { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public List<Function> getSubFunctions() { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public double call(double[] inputs, int j) { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public double derive(double[] inputs, int index, int j) { throw new IllegalAccessError("NOT PART OF THE TEST"); }
    @Override public double derive(double[] inputs, int index) { throw new IllegalAccessError("NOT PART OF THE TEST"); }
}
