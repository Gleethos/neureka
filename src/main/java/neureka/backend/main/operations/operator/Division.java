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
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementWise;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.algorithms.Scalarization;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;
import neureka.devices.Device;
import neureka.ndim.NDimensional;

import java.util.Arrays;


public class Division extends AbstractOperation
{
    public Division()
    {
        super(
                new OperationBuilder()
                        .identifier(         "divide"   )
                        .operator(         "/"        )
                        .arity(            -1         )
                        .isOperator(       true       )
                        .isIndexer(        false      )
                        .isDifferentiable( true       )
                        .isInline(         false      )
        );

        //_____________________
        // DEFAULT OPERATION :

        setAlgorithm(
            BiElementWise.class,
            new BiElementWise(ElemWiseUtil::forDivisionsOrModuli)
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        //________________
        // BROADCASTING :

        setAlgorithm(
                Broadcast.class,
                new Broadcast( ElemWiseUtil::forDivisionsOrModuli )
                .setAutogradModeFor(
                    call -> call
                            .validate().allNotNullHaveSame(NDimensional::shape)
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

        //___________________________
        // TENSOR SCALAR OPERATION :

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor( call -> SuitabilityPredicate.BAD )
            .setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD )
            .setDeviceExecution( (call, callback) -> ElemWiseUtil.forDivisionsOrModuli(call, callback) )
            .buildFunAlgorithm()
        );
    }

    @Override
    public Result execute( Function caller, ExecutionCall<?> call )
    {
        caller = reducePairwise(caller);

        int d = call.getDerivativeIndex();
        if ( !caller.isFlat() ) {
            if ( d < 0 ) {
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( caller, call.withArgs(Arg.DerivIdx.of(-1)) );
                Arrays.stream(flatCall.inputs()).forEach( t -> t.getMut().setIsIntermediate(false) );
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                return super.execute( flat, flatCall );
            }
        }
        if ( d >= 0 ) {
            if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                throw new IllegalArgumentException("The shapes of the operands of the division operation must be equal! (when deriving nested functions)");

            // So here we assume that there are only two sub-functions: a/b

            Function noAd = Function.of( caller.toString(), false );
            Function a = noAd.getSubFunctions().get(0);
            Function b = noAd.getSubFunctions().get(1);
            boolean deriveA = a.dependsOn(d);
            boolean deriveB = b.dependsOn(d);

            if ( !deriveA && !deriveB ) return super.execute( caller, call );

            Tsr<?> bResult = b.call((Call) call.withArgs(Arg.DerivIdx.of(-1)));
            Tsr<?> derivOfA = null;
            if ( deriveA ) {
                Function div = Neureka.get().backend().getFunction().div();
                // This is simple, we just derive the first sub-function and multiply it with the inverse of the second sub-function:
                Tsr<?> aDeriv = a.call((Call)call);
                derivOfA = div.call((Tsr<Object>)aDeriv, (Tsr<Object>)bResult);
            }
            if ( !deriveB && deriveA )
                return Result.of(derivOfA.getMut().setIsIntermediate(true));

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
                    return Result.of(derivOfB.getMut().setIsIntermediate(true));
                else {
                    Function add = Neureka.get().backend().getFunction().add();
                    return Result.of( add.call((Tsr<Object>)derivOfA, (Tsr<Object>)derivOfB).getMut().setIsIntermediate(true) );
                }
            }
        }

        return super.execute( caller, call );
    }

    private Function reducePairwise(Function f) {
        if ( f.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a/b/c/d...
                However, this is how it is really executed:  ((((a/b)/c)/d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = f.getSubFunctions().get(0);
            for ( int i = 1; i < f.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " / " + f.getSubFunctions().get(i), true );

            f = nested;
        }
        return f;
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return _asDerivative( children, derivationIndex, children.length - 1 );
    }

    private String _asDerivative( Function[] children, int d, int index ) {
        if ( d >= 0 ) {
            if ( index <= 0 ) return children[ 0 ].getDerivative( d ).toString();
            else {
                String first = ( children[ index - 1 ].dependsOn( d ) )
                        ? "(" + _asDerivative( children, d, index - 1 )+ " / " + children[ index ]  + " )"
                        : "";

                if ( !children[ index ].dependsOn(d) ) return first;
                String s = children[ index - 1 ].toString();
                if ( s.equals("0.0") ) return first;

                return first +
                        " - ((" + // The second expression is the inner derivative (current index)! (inner times outer...)
                            s + " * " + children[ index ].getDerivative(d) +
                        ") / ( "
                            + children[ index ] + "**2 " +
                        ") )";
            }
        } else {
            if ( index <= 0 ) return children[ 0 ].toString();
            else
                return _asDerivative( children, -1, index - 1 ) + " / " + children[ index ].toString();
        }
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result /= current;
            }
            return result;
        } else {
            double u, ud, v, vd;
            u = src[ 0 ].call( inputs, j );
            ud = src[ 0 ].derive( inputs, d, j );
            for ( int i = 0; i < src.length - 1; i++ ) {
                v = src[ i + 1 ].call( inputs, j );
                vd = src[ i + 1 ].derive( inputs, d, j );
                ud = (ud * v - u * vd) / Math.pow(v, 2);
                u /= v;
            }
            return ud;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result /= current;
            }
            return result;
        } else {
            double derivative;
            double tempVar = src[ 0 ].call( inputs );
            derivative = src[ 0 ].derive( inputs, d );

            for ( int i = 0; i < src.length - 1; i++ ) {
                double u, ud, v, vd;
                v = src[ i + 1 ].call( inputs );
                vd = src[ i + 1 ].derive( inputs, d );
                u = tempVar;
                ud = derivative;
                derivative = ( ud * v - u * vd ) / Math.pow(v, 2);
                tempVar /= v;
            }
            return derivative;
        }
    }




}
