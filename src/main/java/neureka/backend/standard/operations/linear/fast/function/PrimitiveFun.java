/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

public final class PrimitiveFun extends FunctionSet<Double> {

    @FunctionalInterface
    public interface Binary extends BinaryFunction<Double> {

        default Double invoke(final Double arg1, final Double arg2) {
            return Double.valueOf(this.invoke(arg1.doubleValue(), arg2.doubleValue()));
        }

    }

    @FunctionalInterface
    public interface Unary extends UnaryFunction<Double> {

        default Double invoke(final Double arg) {
            return Double.valueOf(this.invoke(arg.doubleValue()));
        }

    }

    private static final PrimitiveFun SET = new PrimitiveFun();

    public static PrimitiveFun getSet() {
        return SET;
    }

    private PrimitiveFun() {
        super();
    }

    @Override
    public BinaryFunction<Double> addition() {
        return FunMath.ADD;
    }

    @Override
    public BinaryFunction<Double> division() {
        return FunMath.DIVIDE;
    }

    @Override
    public BinaryFunction<Double> multiplication() {
        return FunMath.MULTIPLY;
    }

    @Override
    public BinaryFunction<Double> subtraction() {
        return FunMath.SUBTRACT;
    }

}
