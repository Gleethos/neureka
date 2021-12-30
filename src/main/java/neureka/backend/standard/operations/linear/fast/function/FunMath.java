/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

public abstract class FunMath {

    public static final PrimitiveFun.Binary ADD = (arg1, arg2) -> arg1 + arg2;
    public static final PrimitiveFun.Binary DIVIDE = (arg1, arg2) -> arg1 / arg2;
    public static final PrimitiveFun.Binary MULTIPLY = (arg1, arg2) -> arg1 * arg2;
    public static final PrimitiveFun.Unary NEGATE = arg -> -arg;
    public static final PrimitiveFun.Binary SUBTRACT = (arg1, arg2) -> arg1 - arg2;

}
