package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.Activation;
import neureka.backend.main.algorithms.Scalarization;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.calculus.assembly.FunctionParser;

public class AssignLeft extends AbstractOperation
{
    public AssignLeft() {
        super(
            new OperationBuilder()
                .identifier(       "left_inline"  )
                .operator(         "<"            )
                .arity(            -2             )
                .isOperator(       true           )
                .isIndexer(        false          )
                .isDifferentiable( false          )
                .isInline(         true           )
        );

        setAlgorithm(
            Scalarization.class,
            new Scalarization()
            .setIsSuitableFor(
               call -> {
                   if ( call.arity() > 3 )
                       throw new IllegalArgumentException("AssignLeft operation only supports up to 3 arguments!");
                   if ( call.arity() < 2 )
                       throw new IllegalArgumentException("AssignLeft operation needs at least 2 arguments!");

                   int offset = call.arity() - 1;
                   if ( call.input( offset ).isVirtual() || call.input( offset ).size() == 1 )
                       return  call.validate()
                                       .allNotNull( t -> t.getDataType().typeClassImplements(Object.class) )
                                       //.allNotNull( Tsr::isVirtual )
                                       .tensors( tensors -> tensors.length == 2 || tensors.length == 3 )
                                       .suitabilityIfValid(SuitabilityPredicate.PERFECT);
                   else
                       return SuitabilityPredicate.UNSUITABLE;
               }
            )
            .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
            .setExecution( (caller, call) -> {
                Tsr<?> t = AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, null );
                t.getMut().incrementVersion(call);
                return Result.of(t);
            })
            .setCallPreparation(
                call -> {
                    int offset = ( call.input( 0 ) == null ? 1 : 0 );
                    call.input( offset ).getMut().setIsVirtual( false );
                    return
                        ExecutionCall.of( call.input( offset ), call.input( offset + 1 ) )
                                .andArgs(Arg.DerivIdx.of(-1))
                                .running(this)
                                .on( call.getDevice() );
                }
            )
            .buildFunAlgorithm()
        );

        setAlgorithm(
            new Activation()
            .setIsSuitableFor(
                call -> call.validate()
                        .allNotNull( t -> t.getDataType().typeClassImplements(Object.class) )
                        //.allNotNull( t -> !t.isVirtual() )
                        .tensors( tensors -> tensors.length == 2 || tensors.length == 3 )
                        .suitabilityIfValid(SuitabilityPredicate.EXCELLENT)
            )
            .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
            .setExecution( (caller, call) -> {
                Tsr<?> t = AbstractDeviceAlgorithm.executeDeviceAlgorithm( call, null );
                t.getMut().incrementVersion(call);
                return Result.of(t);
            })
            .setCallPreparation(
                    call -> {
                        int offset = ( call.input( 0 ) == null ? 1 : 0 );
                        return ExecutionCall.of( call.input(offset), call.input(1+offset) )
                                .running(Neureka.get().backend().getOperation("idy"))
                                .on( call.getDevice() );
                    }
            )
            .buildFunAlgorithm()
        );
    }


    @Override
    public Result execute( Function caller, ExecutionCall<?> call )
    {
        if ( call.getDerivativeIndex() >= 0 )
            throw new IllegalArgumentException("Assignment does not support autograd!");

        caller = reducePairwise(caller);
        ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( caller, call.withArgs(Arg.DerivIdx.of(-1)) );
        for (Tsr<?> t : call.inputs()) t.getMut().setIsIntermediate(false);
        Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), false );
        return super.execute( flat, flatCall );
    }

    private Function reducePairwise(Function f) {
        if ( f.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a <- b <- c <- d...
                However, this is how it is really executed:  (a**(b**(c**(d**..))))
                ...so let's create a function that is nested like the above:
            */
            Function nested = f.getSubFunctions().get(f.getSubFunctions().size()-1);
            for ( int i = f.getSubFunctions().size()-2; i >= 0; i-- )
                nested = Function.of( f.getSubFunctions().get(i) + " <- " + nested, true );

            f = nested;
        }
        return f;
    }


    @Override
    public String stringify( String[] children ) {
        StringBuilder reconstructed = new StringBuilder();
        for ( int i = 0; i < children.length; ++i ) {
            reconstructed.append( children[ i ] );
            if ( i < children.length - 1 ) reconstructed.append(" <- ");
        }
        return "(" + reconstructed + ")";
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        int right = src.length - 1;
        return d >= 0 ? src[ right ].derive( inputs, d, j ) : src[ right ].call( inputs, j );
    }
}
