package neureka.backend.main.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.main.internal.AlgoUtil;
import neureka.backend.main.internal.CallExecutor;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *  Methods inside this utility class execute only some {@link ExecutionCall} arguments
 *  in groups if their total number exceeds the arity of an operation.
 *  
 */
public class ElemWiseUtil
{
    private static final Logger _LOG = LoggerFactory.getLogger( ElemWiseUtil.class );

    public static Tsr<?> forMultiplications(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        call = call.withInputs(call.inputs()); // Let's make sure we prevent any side effects.
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Tsr<?> result = null;
        if ( call.arity() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ call.input( 0 ), call.input( 1 ), call.input( 2 ) };
                result = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction ).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                                );
                call = call.withInputAt( 0, result );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                result = recursiveExecutor.execute(
                        ExecutionCall.of( reduction ).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                call = call.withInputAt( 0, result );
            } else {
                Tsr<?>[] reduction = Operation.Utility.without(call.inputs(), 1+d);
                if ( reduction.length > 2 ) {
                    reduction[ 0 ] = ( reduction[ 0 ] == null ) ? call.input( 1 ).deepCopy().getUnsafe().setIsIntermediate( true ) : reduction[ 0 ];
                    result = recursiveExecutor.execute(
                            ExecutionCall.of( reduction )
                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                            .running( Neureka.get().backend().getOperation("*") )
                                            .on( device )
                    );
                    call = call.withInputAt( 0, result );
                }
                else
                    call = call.withInputAt( 0, reduction[ 1 ] );
            }
            if ( result == null ) return AlgoUtil.executeDeviceAlgorithm( call, null );
            return result;
        } 
        else
            return AlgoUtil.executeDeviceAlgorithm( call, null );

    }

    @Contract( pure = true )
    public static Tsr<?> forDivisionsOrModuli(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        call = call.withInputs(call.inputs().clone()); // Let's make sure we prevent any side effects.
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );

        Tsr<?> result = null;
        if ( call.arity() > 3 )
        {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                result = recursiveExecutor.execute( call.withInputs( reduction ) );
                call.setInput( 0, result );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                result = recursiveExecutor.execute(
                                    call.withInputs(reduction)
                            );
                call.setInput( 0, result );
            } else {
                Tsr<?> a;
                if ( d > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset(call.inputs(), 1, 1, d+1);
                    reduction[ 0 ] = call.input( 1 ).deepCopy().getUnsafe().setIsIntermediate( true );
                    result = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                    );
                    a = result;
                }
                else if ( d == 1 ) a = call.input( 1 );
                else a = newTsrLike( call.input( 1 ), 1.0 );
                Tsr<?> b;
                if ( call.arity() -  d - 2  > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset( call.inputs(), 2, d+2, call.arity()-(d+2) );
                    reduction[ 1 ] = newTsrLike( call.input( 1 ), 1.0 );
                    reduction[ 0 ] = reduction[ 1 ];
                    result = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                    b = result;
                }
                else b = newTsrLike( call.input( 1 ), 1.0 );

                result = recursiveExecutor.execute(
                                        ExecutionCall.of( call.input( 0 ), a, b )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("*") )
                                                        .on( device )
                                );
                result = recursiveExecutor.execute(
                                        ExecutionCall.of( result, call.input( 0 ), call.input( d + 1 ) )
                                                        .andArgs(Arg.DerivIdx.of(1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                if ( d == 0 ) a.getUnsafe().delete();
                b.getUnsafe().delete();
            }
        }
        if ( result == null ) return AlgoUtil.executeDeviceAlgorithm( call, null );
        return result;
    }

    @Contract( pure = true )
    public static Tsr<?> forAdditions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        return _forAdditionsOrSubtractions(call, recursiveExecutor, true);
    }

    @Contract( pure = true )
    public static Tsr<?> forSubtractions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor
    ) {
        return _forAdditionsOrSubtractions(call, recursiveExecutor, false);
    }

    @Contract( pure = true )
    private static Tsr<?> _forAdditionsOrSubtractions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor,
            boolean thisIsForAddition
    ) {
        call = call.withInputs(call.inputs().clone()); // Let's make sure there are no side effects!
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> result = null;
        if ( call.arity() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                result = recursiveExecutor.execute(
                                    ExecutionCall.of( reduction )
                                                    .andArgs(Arg.DerivIdx.of(d))
                                                    .running(operation)
                                                    .on(device)
                            );
                call = call.withInputAt(0, result );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                result = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(d))
                                                        .running(operation)
                                                        .on(device)
                                );
                call = call.withInputAt(0, result );
            }
            else
                call = call.withInputAt(0,
                        call.input( 1 ).deepCopy()
                           .getUnsafe()
                           .setIsIntermediate( true )
                           .setItems( d == 0 || thisIsForAddition ? 1f : -1f )
                    );
        }
        if ( result == null ) return AlgoUtil.executeDeviceAlgorithm( call, null );
        return result;
    }


    public static <V> Tsr<V> newTsrLike( Tsr<V> template, double value ) {
        return newTsrLike(
            template.itemClass(),
            template.getNDConf().shape(),
            template.isOutsourced(),
            template.get( Device.class ),
            value
        );
    }

    public static <V> Tsr<V> newTsrLike(
        Class<V> type, int[] shape, boolean isOutsourced, Device<Object> device, double value
    ) {
        Tsr<V> t = Tsr.of( type, shape, value )
                        .getUnsafe()
                        .setIsIntermediate( true );
        t.setIsVirtual( false );
        t.setItems( value );
        try {
            if ( isOutsourced ) device.store( t );
        } catch ( Exception exception ) {
            _LOG.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
            throw exception;
        }
        return t;
    }

}
