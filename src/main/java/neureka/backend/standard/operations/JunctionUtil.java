package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.internal.CallExecutor;
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
public class JunctionUtil
{
    private static final Logger _LOG = LoggerFactory.getLogger( JunctionUtil.class );

    @Contract( pure = true )
    public static Tsr<?> forConvolution(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> alternative = null;
        if ( call.size() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ call.input( 0 ), call.input( 1 ), call.input( 2 ) };
                alternative = recursiveExecutor.execute(
                                    ExecutionCall.of( reduction )
                                                    .andArgs( Arg.DerivIdx.of(d) )
                                                    .running( operation )
                                                    .on(device)
                                );
                call.setInput( 0, alternative );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                alternative = recursiveExecutor.execute(
                                    ExecutionCall.of( reduction )
                                                    .andArgs(Arg.DerivIdx.of(d))
                                                    .running(operation)
                                                    .on(device)
                                );
                call.setInput( 0, alternative );
            }
            return alternative;
        } else {
            if ( call.getOperation().getOperator().equals("x") ) {
                if ( d >= 0 ) {
                    if ( d == 0 ) call.setInput( 0, call.input( 2 ) );
                    else call.setInput( 0, call.input( 1 ) );
                    return call.input( 0 );
                } else {
                    call.rearrangeInputs( 0, 1, 2 );
                }
            } else if ( call.getOperation().getOperator().equals("x"+ ((char) 187)) ) {
                call.rearrangeInputs( 2, 1, 0 );
            }
            return null;
        }
    }

    public static Tsr<?> forMultiplications(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Tsr<?> alternative = null;
        if ( call.size() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ call.input( 0 ), call.input( 1 ), call.input( 2 ) };
                alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction ).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                                );
                call.setInput( 0, alternative );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                alternative = recursiveExecutor.execute(
                        ExecutionCall.of( reduction ).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                call.setInput( 0, alternative );
            } else {
                Tsr<?>[] reduction = Operation.Utility.without(call.inputs(), 1+d);
                if ( reduction.length > 2 ) {
                    reduction[ 0 ] = ( reduction[ 0 ] == null ) ? call.input( 1 ).clone().getUnsafe().setIsIntermediate( true ) : reduction[ 0 ];
                    alternative = recursiveExecutor.execute(
                            ExecutionCall.of( reduction )
                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                            .running( Neureka.get().backend().getOperation("*") )
                                            .on( device )
                    );
                    call.setInput( 0, alternative );
                }
                else
                    call.setInput( 0, reduction[ 1 ] );
            }
            return alternative;
        } 
        else
            return null;

    }

    @Contract( pure = true )
    public static Tsr<?> forDivisionsOrModuli(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor recursiveExecutor // This will indirectly be a recursive call!
    ) {
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );

        Tsr<?> alternative = null;
        if ( call.size() > 3 )
        {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                alternative = recursiveExecutor.execute( call.withTensors( reduction ) );
                call.setInput( 0, alternative );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                alternative = recursiveExecutor.execute(
                                    call.withTensors(reduction)
                            );
                call.setInput( 0, alternative );
            } else {
                Tsr<?> a;
                if ( d > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset(call.inputs(), 1, 1, d+1);
                    reduction[ 0 ] = call.input( 1 ).clone().getUnsafe().setIsIntermediate( true );
                    alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                    );
                    a = alternative;
                }
                else if ( d == 1 ) a = call.input( 1 );
                else a = newTsrLike( (Tsr<Number>) call.input( 1 ), 1.0 );
                Tsr<?> b;
                if ( call.size() -  d - 2  > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset( call.inputs(), 2, d+2, call.size()-(d+2) );
                    reduction[ 1 ] = newTsrLike( (Tsr<Number>) call.input( 1 ), 1.0 );
                    reduction[ 0 ] = reduction[ 1 ];
                    alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                    b = alternative;
                }
                else b = newTsrLike( (Tsr<Number>) call.input( 1 ), 1.0 );

                alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( call.input( 0 ), a, b )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("*") )
                                                        .on( device )
                                );
                alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( alternative, call.input( 0 ), call.input( d + 1 ) )
                                                        .andArgs(Arg.DerivIdx.of(1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                if ( d == 0 ) a.getUnsafe().delete();
                b.getUnsafe().delete();
            }
        }
        return alternative;
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
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> alternative = null;
        if ( call.size() > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{call.input( 0 ), call.input( 1 ), call.input( 2 )};
                alternative = recursiveExecutor.execute(
                                    ExecutionCall.of( reduction )
                                                    .andArgs(Arg.DerivIdx.of(d))
                                                    .running(operation)
                                                    .on(device)
                            );
                call.setInput( 0, alternative );

                reduction = Operation.Utility.offsetted(call.inputs(), 1);
                alternative = recursiveExecutor.execute(
                                        ExecutionCall.of( reduction )
                                                        .andArgs(Arg.DerivIdx.of(d))
                                                        .running(operation)
                                                        .on(device)
                                );
                call.setInput( 0, alternative );
            }
            else
                call.setInput( 0,
                        call.input( 1 ).clone()
                           .getUnsafe()
                           .setIsIntermediate( true )
                           .setValue( d == 0 || thisIsForAddition ? 1f : -1f )
                    );
        }
        return alternative;
    }


    public static <V> Tsr<V> newTsrLike( Tsr<V> template, double value ) {
        //Tsr<V> t = (Tsr<V>) Tsr.like( (Tsr<Number>) template ).all( value );
        //t.setIsVirtual(false);
        Tsr<V> t = Tsr.of( template.getValueClass(), template.getNDConf().shape(), value )
                        .getUnsafe()
                        .setIsIntermediate( true );
        t.setIsVirtual( false );
        t.setValue( value );
        try {
            if ( template.isOutsourced() ) ( (Device<Object>) template.get( Device.class ) ).store( t );
        } catch ( Exception exception ) {
            _LOG.error( "Failed storing a newly created tensor from a template tensor to its host device.", exception );
            throw exception;
        }
        return t;
    }

}
