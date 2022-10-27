package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Call;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.ndim.NDimensional;

import java.util.Arrays;

public class Modulo extends AbstractOperation
{
    public Modulo()
    {
        super(
            new OperationBuilder()
                .identifier(       "modulo"    )
                .operator(         "%"         )
                .arity(            -1          )
                .isOperator(       true        )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            BiElementWise.class,
            new BiElementWise()
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Broadcast.class,
            new Broadcast()
            .setAutogradModeFor(
                call -> call.validate()
                        .allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setSupplyADActionFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tsr<?> ctxDerivative = (Tsr<?>) call.getValOf(Arg.Derivative.class);
                    Function mul = Neureka.get().backend().getFunction().mul();
                    if ( ctxDerivative != null ) {
                        return ADAction.of( target -> mul.execute( target.error(), ctxDerivative ) );
                    }
                    int d = call.getDerivativeIndex();
                    Tsr<?> derivative = f.executeDerive( call.inputs(), d );
                    return ADAction.of( target -> mul.execute( target.error(), derivative ) );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
            .setAutogradModeFor(
                call -> call.validate()
                        .allNotNullHaveSame(NDimensional::shape)
                        .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                        .orElse(AutoDiffMode.BACKWARD_ONLY)
            )
            .setExecution( (caller, call) -> Result.of(AbstractDeviceAlgorithm.executeFor(caller, call, AbstractDeviceAlgorithm::executeDeviceAlgorithm)).withAutoDiff( FallbackAlgorithm::ADAction ))
            .buildFunAlgorithm()
        );
    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        Function reducedCaller = reducePairwise(caller);

        int d = call.getDerivativeIndex();
        if ( !reducedCaller.isFlat() ) {
            if ( d < 0 ) {
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
                Arrays.stream(flatCall.inputs()).forEach(t -> t.mut().setIsIntermediate(false) );
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                return super.execute( flat, flatCall );
            }
        }
        if ( d >= 0 ) {
            if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                throw new IllegalArgumentException("The shapes of the operands of the division operation must be equal! (when deriving nested functions)");

            // So here we assume that there are only two sub-functions: a/b

            Function noAd = Function.of( reducedCaller.toString(), false );
            Function a = noAd.getSubFunctions().get(0);
            Function b = noAd.getSubFunctions().get(1);
            boolean deriveA = a.dependsOn(d);
            boolean deriveB = b.dependsOn(d);

            if ( !deriveA && !deriveB ) return super.execute( reducedCaller, call );

            Tsr<?> bResult = b.call((Call) call.withArgs(Arg.DerivIdx.of(-1)));
            Tsr<?> derivOfA = null;
            if ( deriveA ) {
                Function div = Neureka.get().backend().getFunction().div();
                // This is simple, we just derive the first sub-function and multiply it with the inverse of the second sub-function:
                Tsr<?> aDeriv = a.call((Call)call);
                derivOfA = div.call((Tsr<Object>)aDeriv, (Tsr<Object>)bResult);
            }
            if ( !deriveB && deriveA )
                return Result.of(derivOfA.mut().setIsIntermediate(true));

            Tsr<?> aResult = a.call((Call)call.withArgs(Arg.DerivIdx.of(-1)));
            if ( deriveB ) {
                Function mul = Neureka.get().backend().getFunction().mul();
                Tsr<?> innerDerivB = b.call((Call)call);
                // So we have something like this: a/b, where we want to derive b.
                // This is how it is really executed:  (a/b) = (a * (1/b))
                // So we can derive b and then later on add the derivative of 'a' to it (if it must be derived).
                // The derivative of 1/b is -1/b^2
                // Let's derive b:
                Function derive = Function.of("-I[0] / (I[1] ** 2)", false);
                Tsr<?> derivOfB = derive.call( (Tsr<Object>)innerDerivB, (Tsr<Object>)bResult );
                derivOfB = mul.call((Tsr<Object>)aResult, (Tsr<Object>)derivOfB);
                if ( !deriveA )
                    return Result.of(derivOfB.mut().setIsIntermediate(true));
                else {
                    Function add = Neureka.get().backend().getFunction().add();
                    return Result.of( add.call((Tsr<Object>)derivOfA, (Tsr<Object>)derivOfB).mut().setIsIntermediate(true) );
                }
            }
        }

        return super.execute( reducedCaller, call );
    }

    private Function reducePairwise( final Function fun ) {
        Function reduced = fun;
        if ( reduced.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a%b%c%d...
                However, this is how it is really executed:  ((((a%b)%c)%d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = reduced.getSubFunctions().get(0);
            for ( int i = 1; i < reduced.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " % " + reduced.getSubFunctions().get(i), true );

            reduced = nested;
        }
        return reduced;
    }

    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result %= current;
            }
            return result;
        }
        else return src[ 0 ].derive( inputs, d );
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return children[ 0 ].getDerivative(derivationIndex).toString();
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result %= current;
            }
            return result;
        }
        else
            return src[ 0 ].derive( inputs, d, j );
    }

}
