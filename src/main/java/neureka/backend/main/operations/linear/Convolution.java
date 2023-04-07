package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.Shape;
import neureka.Tsr;
import neureka.autograd.ADAction;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.NDConvolution;
import neureka.backend.main.operations.ConvUtil;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;
import neureka.devices.Device;

public class Convolution extends AbstractOperation
{
    public Convolution()
    {
        super(
            new OperationBuilder()
                .identifier(       "mul_conv"  )
                .operator(         "x"         )
                .arity(            2           )
                .isOperator(       true        )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            NDConvolution.class,
            new NDConvolution()
            .setAutogradModeFor( call -> {
                if ( call.getOperation().supports( NDConvolution.class ) ) return AutoDiffMode.BACKWARD_ONLY;
                Tsr<?> last = null;
                for ( Tsr<?> t : call.inputs() ) {
                    if ( last != null && !last.shape().equals(t.shape()) ) return AutoDiffMode.BACKWARD_ONLY;
                    last = t; // Note: shapes are cached!
                }
                return AutoDiffMode.FORWARD_AND_BACKWARD;
            })
            .setExecution(
                (outerCaller, outerCall) ->
                    Result.of(AbstractDeviceAlgorithm.prepareAndExecute(
                        outerCall,
                        call ->
                                AbstractDeviceAlgorithm.executeDeviceAlgorithm(
                                        call
                                )
                    ))
                    .withAutoDiff(( Function f, ExecutionCall<? extends Device<?>> adCall ) ->
                    {
                        int d = adCall.getDerivativeIndex();
                        Function deConv = new FunctionParser( Neureka.get().backend() ).parse(
                                "I[ 0 ] x>> I[ 1 ] x>> I[ 2 ]",
                                false
                        );
                        Tsr<?> derivative = f.derive( (Tsr[]) adCall.inputs(), d );
                        assert d >= 0 && d <= 1;
                        assert derivative != null;
                        assert deConv != null;
                        assert adCall.arity() >= 2 && adCall.arity() <= 3;
                        // Now we need to remember the shape of the input which is targeted for back prop.
                        Shape shape = Shape.of(adCall.input( adCall.arity() > 2 ? d + 1 : d ).getNDConf().shape());
                        Number zero;
                        if ( derivative.getItemType() == Double.class         ) zero = 0d;
                        else if ( derivative.getItemType() == Float.class     ) zero = 0f;
                        else if ( derivative.getItemType() == Integer.class   ) zero = 0;
                        else if ( derivative.getItemType() == Long.class      ) zero = 0L;
                        else if ( derivative.getItemType() == Short.class     ) zero = (short) 0;
                        else if ( derivative.getItemType() == Byte.class      ) zero = (byte) 0;
                        else {
                            zero = null;
                            throw new RuntimeException("Unsupported item type for convolution derivative: " + derivative.getItemType());
                        }
                        // This is because it will be the shape of the output to the de-convolution!
                        return ADAction.of( target ->
                                deConv.execute(
                                        target.error(),
                                        derivative,
                                        Tsr.of(shape, zero).mut().setIsIntermediate( false )
                                )
                        );
                    })
            )
            .setCallPreparation(
                 call -> {
                     if ( call.arity() <= 2 ) call = call.withAddedInputAt( 0, null );
                     Device<Number> device = call.getDeviceFor(Number.class);
                     int[] shp = ConvUtil.shapeOfCon(call.input( 1 ).getNDConf().shape(), call.input( 2 ).getNDConf().shape());
                     Tsr<Number> output = (Tsr<Number>) Tsr.of( call.input(1).getItemType(), shp, 0 )
                                                             .mut()
                                                             .setIsIntermediate( true );
                     output.mut().setIsVirtual( false );
                     //device.store( output );//Todo: find out why this causes problems
                     return call.withInputAt( 0, output );
                 }
            )
            .buildFunAlgorithm()
        );

    }


    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        if ( !caller.isFlat() ) {
            Function reducedCaller = reducePairwise(caller);
            ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
            Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
            for ( Tsr<?> t : flatCall.inputs() ) if ( t != null ) t.mut().setIsIntermediate(false);
            return this.execute( flat, flatCall );
        }
        if ( call.getDerivativeIndex() >= 0 ) {
            int d = call.getDerivativeIndex();
            /*
                In autograd convolution is similar to matrix multiplication.
                If the derivative index is 0 then the second operand is used for backward broadcasting.
                If the derivative index is 1 then the first operand is used for backward broadcasting.
             */
            return Result.of( call.input( d == 0 ? 1 : 0 ) );
        }
        Function reducedCaller = reducePairwise(caller);
        ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
        Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
        for ( Tsr<?> t : flatCall.inputs() ) if ( t != null ) t.mut().setIsIntermediate(false);
        return super.execute( flat, flatCall );
    }

    private Function reducePairwise( final Function fun ) {
        Function reduced = fun;
        if ( reduced.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a x b x c x d...
                However, this is how it is really executed:  ((((a x b) x c) x d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = reduced.getSubFunctions().get(0);
            for ( int i = 1; i < reduced.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " x " + reduced.getSubFunctions().get(i), true );

            reduced = nested;
        }
        return reduced;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
