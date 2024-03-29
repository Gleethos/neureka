package neureka.backend.main.operations.operator;

import neureka.Neureka;
import neureka.Tensor;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.BiElementwise;
import neureka.backend.main.algorithms.BiScalarBroadcast;
import neureka.backend.main.algorithms.Broadcast;
import neureka.backend.main.operations.ElemWiseUtil;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;
import neureka.devices.Device;
import neureka.ndim.NDimensional;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Addition extends AbstractOperation {

    public Addition()
    {
        super (
            new OperationBuilder()
            .identifier(       "add"      )
            .operator(         "+"        )
            .arity(            -1         )
            .isOperator(       true       )
            .isIndexer(        false      )
            .isDifferentiable( true       )
            .isInline(         false      )
        );

        setAlgorithm(
            new BiElementwise()
            .setExecution(
                (outerCaller, outerCall) ->
                    Result.of(AbstractDeviceAlgorithm.prepareAndExecute(
                        outerCall,
                        innerCall -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( innerCall )
                    ))
            )
            .setSupplyADActionFor( getDefaultAlgorithm() )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new Broadcast()
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setSupplyADActionFor(
                ( Function f, ExecutionCall<? extends Device<?>> call ) ->
                {
                    if ( call.autogradMode().allowsForward() )
                        throw new IllegalArgumentException("Broadcast implementation does not support forward-AD!");
                    Tensor<?> ctxDerivative = (Tensor<?>) call.getValOf(Arg.Derivative.class);
                    assert ctxDerivative == null;
                    return _autogradBroadcast( call );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new BiScalarBroadcast()
            .setExecution(
                (iniCaller, iniCall) ->
                    Result.of(AbstractDeviceAlgorithm.prepareAndExecute( iniCall, AbstractDeviceAlgorithm::executeDeviceAlgorithm))
                        .withAutoDiff( (caller, call) -> {
                            if ( call.getDerivativeIndex() >= 0 && call.arity() >= 2 ) {
                                int offset = call.input(0) == null ? 1 : 0;
                                boolean thisIsBroadcasting = !call.input(offset).shape().equals(call.input(offset + 1).shape());
                                if ( thisIsBroadcasting )
                                    return _autogradBroadcast( call );
                            }
                            return FallbackAlgorithm.ADAction(caller, call);
                        } )
            )
            .buildFunAlgorithm()
        );
    }

    private ADAction _autogradBroadcast(ExecutionCall<? extends Device<?>> call) {
        int d = call.getDerivativeIndex();
        Tensor<?> derivative = ElemWiseUtil.newTensorLike(call.input( d==0?1:0 ), 0);
        Tensor<?> toBeDerived = ElemWiseUtil.newTensorLike(call.input( d ), 0);
        Device device = call.getDeviceFor(Number.class);
        return ADAction.of(
                target ->
                        this.getAlgorithm( Broadcast.class )
                                .getImplementationFor( device )
                                .run(
                                        ExecutionCall.of(
                                                        toBeDerived.mut().setIsVirtual(false),
                                                        derivative,
                                                        target.error()
                                                )
                                                .andArgs( Arg.DerivIdx.of(d) )
                                                .running( this )
                                                .on( device )
                                )
                );
    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        int d = call.getDerivativeIndex();
        if ( caller.isFlat() ) {
            if ( d >= 0 && call.arity() >= 2 ) {
                int offset = call.input( 0 ) == null ? 1 : 0;
                boolean thisIsBroadcasting = !call.input( offset ).shape().equals( call.input( offset + 1 ).shape() );
                if ( thisIsBroadcasting ) {
                    /*
                        In autograd broadcasting is similar to matrix multiplication.
                        If the derivative index is 0 then the second operand is used for backward broadcasting.
                        If the derivative index is 1 then the first operand is used for backward broadcasting.
                     */
                    return Result.of( call.input( d == 0 ? 1 : 0 ) );
                }
                int j = call.getValOf(Arg.VarIdx.class);
                Tensor<?> template = call.input( offset + d );
                long dependencies = caller.getSubFunctions()
                                            .stream()
                                            .filter( f -> f.dependsOn(d) && j < 0 || (j == d && f.dependsOn(d)))
                                            .count();

                Tensor<?> derivative = Tensor.like((Tensor<Number>) template).all(dependencies);
                return Result.of(derivative.mut().setIsIntermediate(true));
            }
        } else {
            if ( d < 0 ) {
                Function reducedCaller = reducePairwise(caller);
                ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
                Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
                return super.execute( flat, flatCall );
            } else {
                if ( !call.validate().allNotNullHaveSame(NDimensional::shape).isValid() )
                    throw new IllegalArgumentException("The shapes of the operands of the addition operation must be equal! (when deriving nested functions)");

                int[] toBeDerived = IntStream.range(0,caller.getSubFunctions().size())
                                            .filter( i -> caller.getSubFunctions().get(i).dependsOn(d) )
                                            .toArray();

                Tensor[] results = new Tensor[ toBeDerived.length ];
                for ( int i = 0; i < results.length; i++ ) {
                    Function noAD = Function.of( caller.getSubFunctions().get( toBeDerived[i] ).toString(), false );
                    Tensor<?> deriv = noAD.execute( noAD.getOperation() == null ? call : call.withOperation(noAD.getOperation()) );
                    results[ i ] = deriv;
                }
                if ( results.length == 1 ) return Result.of( results[0] );
                Function addAll = new FunctionParser(Neureka.get().backend()).parse(Neureka.get().backend().getOperation("+"), results.length, false);
                return addAll.getOperation().execute(addAll, call.withInputs(results).withArgs(Arg.DerivIdx.of(-1)));
            }
        }
        Function reducedCaller = reducePairwise(caller);
        ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
        Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
        Result r = super.execute( flat, flatCall );
        //for ( int i = 0; i < flatCall.inputs().length; i++ )
        //    _deleteIfNotIn(call.inputs(), flatCall.input(i)); // TODO: Make it possible to delete more stuff
        return r;
    }

    private Function reducePairwise( final Function fun ) {
        Function reduced = fun;
        if ( reduced.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a+b+c+d...
                However, this is how it is really executed:  ((((a+b)+c)+d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = reduced.getSubFunctions().get(0);
            for ( int i = 1; i < reduced.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " + " + reduced.getSubFunctions().get(i), true );

            reduced = nested;
        }
        return reduced;
    }


    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        return Arrays.stream( children )
                    .filter( child -> child.dependsOn(derivationIndex) )
                    .map( child -> child.getDerivative(derivationIndex) )
                    .map( Object::toString )
                    .collect( Collectors.joining( " "+getOperator()+" " ) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs, j );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs, j );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive(inputs, d, j);

            return derivative;
        }
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 ) {
            double result = src[ 0 ].call( inputs );
            for ( int i = 1; i < src.length; i++ ) {
                final double current = src[ i ].call( inputs );
                result += current;
            }
            return result;
        } else {
            double derivative = 0;
            for ( Function function : src )
                derivative += function.derive( inputs, d );

            return derivative;
        }
    }




}
