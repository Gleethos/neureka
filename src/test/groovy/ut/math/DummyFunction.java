package ut.math;

import neureka.Tensor;
import neureka.backend.api.Operation;
import neureka.math.Function;
import neureka.math.args.Args;

import java.util.List;
import java.util.function.BiFunction;

public class DummyFunction implements Function {

    private final BiFunction<Object, Object, Tensor<?>> implementation;

    DummyFunction( BiFunction<Object, Object, Tensor<?>> implementation ) {
        this.implementation = implementation;
    }

    @Override
    public Tensor<?> execute(Args arguments, Tensor<?>... inputs ) {
        return implementation.apply( arguments, inputs.clone() );
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
