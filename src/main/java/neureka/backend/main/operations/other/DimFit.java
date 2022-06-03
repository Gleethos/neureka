package neureka.backend.main.operations.other;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.internal.CalcUtil;
import org.jetbrains.annotations.Contract;

public class DimFit extends AbstractOperation
{
    public DimFit()
    {
        super(
            new OperationBuilder()
                    .setIdentifier(       "dimfit"    )
                    .setOperator(         "dimfit"    )
                    .setArity(            -1          )
                    .setIsOperator(       false       )
                    .setIsIndexer(        false       )
                    .setIsDifferentiable( true        )
                    .setIsInline(         false       )
        );

        setAlgorithm(
            Algorithm
            .withName("reshape")
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tsr<?>[] inputs = CalcUtil.srcActivation(call.inputs(), call.getValOf( Arg.VarIdx.class ), -1, 0, caller.getSubFunctions().toArray(new Function[0]));
                    assert call.getValOf( Arg.DerivIdx.class ) < 0;

                    int largest = -1;
                    int[] shape = null;
                    for ( Tsr<?> t : inputs ) if ( t.rank() > largest ) {
                        largest = t.rank();
                        shape = t.getNDConf().shape();
                    }
                    int prefix = 0;
                    for ( int s : shape ) if ( s == 1 ) prefix++; else break;
                    int postfix = 0;
                    for ( int i = shape.length-1; i>=0; i-- ) if ( shape[ i ] == 1 ) postfix++; else break;

                    int[][] change = new int[inputs.length][];

                    for ( int i=0; i<inputs.length; i++)
                    {
                        if ( inputs[ i ].rank()!=largest)
                        {
                            int[] oldShape = inputs[ i ].getNDConf().shape();
                            int[] newReshape = new int[largest];
                            int padding = largest-oldShape.length;

                            int handle = ( postfix <= prefix )? padding : largest-padding;
                            for ( int ii=0; ii<handle; ii++) newReshape[ ii ]       = ( postfix <= prefix )? -1 : ii;
                            for ( int ii=handle; ii<largest; ii++) newReshape[ ii ] = ( postfix <= prefix )? ii-padding : -1;

                            change[ i ] = newReshape;
                        }
                    }
                    return Result.of(null).withADAction(null);
                }
            )
            .buildFunAlgorithm()
        );
    }

    @Contract( pure = true )
    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if (expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')') {
            return "dimfit" + expression;
        }
        return "dimfit" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}