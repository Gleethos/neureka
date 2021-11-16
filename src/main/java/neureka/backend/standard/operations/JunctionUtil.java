package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.CallExecutor;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import org.jetbrains.annotations.Contract;

public class JunctionUtil
{
    @Contract( pure = true )
    public static Tsr<?> forConvolution(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith
    ) {
        Tsr<?>[] tensors = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> alternative = null;
        if ( tensors.length > 3 ) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ tensors[ 0 ], tensors[ 1 ], tensors[ 2 ] };
                alternative = goDeeperWith.execute(
                                    ExecutionCall.of( reduction )
                                                    .andArgs( Arg.DerivIdx.of(d) )
                                                    .running( operation )
                                                    .on(device)
                                );
                tensors[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tensors, 1);
                alternative = goDeeperWith.execute(
                                    ExecutionCall.of(reduction)
                                                    .andArgs(Arg.DerivIdx.of(d))
                                                    .running(operation)
                                                    .on(device)
                                );
                tensors[ 0 ] = reduction[ 0 ];
            }
            return alternative;
        } else {
            if ( call.getOperation().getOperator().equals("x") ) {
                if ( d >= 0 ) {
                    if ( d == 0 ) tensors[ 0 ] = tensors[ 2 ];
                    else tensors[ 0 ] = tensors[ 1 ];
                    return tensors[ 0 ];
                } else {
                    call.mutateTensors( 0, 1, 2 );
                }
            } else if ( call.getOperation().getOperator().equals("x"+ ((char) 187)) ) {
                call.mutateTensors( 2, 1, 0 );
            }
            return alternative;
        }
    }

    public static Tsr<?> forMultiplications(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith
    ) {
        Tsr<?>[] tsrs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Tsr<?> alternative = null;
        if (tsrs.length > 3) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ] };
                alternative = goDeeperWith.execute(
                        ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.execute(
                        ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                tsrs[ 0 ] = reduction[ 0 ];
            } else {
                Tsr<?>[] reduction = Operation.Utility.without(tsrs, 1+d);
                if ( reduction.length > 2 ) {
                    reduction[ 0 ] = ( reduction[ 0 ] == null ) ? Tsr.Create.newTsrLike(tsrs[ 1 ]) : reduction[ 0 ];
                    alternative = goDeeperWith.execute(
                            ExecutionCall.of(reduction)
                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                            .running( Neureka.get().backend().getOperation("*") )
                                            .on( device )
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else tsrs[ 0 ] = reduction[ 1 ];
            }
            return alternative;
        } else
            return alternative;

    }

    @Contract( pure = true )
    public static Tsr<?> forDivisionsOrModuli(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith
    ) {
        Tsr<?>[] tsrs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation type = call.getOperation();

        Tsr<?> alternative = null;
        if ( tsrs.length > 3 )
        {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                alternative = goDeeperWith.execute(
                        call.withTensors( reduction )
                            );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.execute(
                                    call.withTensors(reduction)
                            );
                tsrs[ 0 ] = reduction[ 0 ];
            } else {
                Tsr<?> a;
                if ( d > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset(tsrs, 1, 1, d+1);
                    reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                    alternative = goDeeperWith.execute(
                                        ExecutionCall.of(reduction)
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                    );
                    a = reduction[ 0 ];
                }
                else if ( d == 1 ) a = tsrs[ 1 ];
                else a = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                Tsr<?> b;
                if ( tsrs.length -  d - 2  > 1 ) {
                    Tsr<?>[] reduction = Operation.Utility.subset(tsrs, 2, d+2, tsrs.length-(d+2));
                    reduction[ 1 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                    reduction[ 0 ] = reduction[ 1 ];
                    alternative = goDeeperWith.execute(
                                        ExecutionCall.of(reduction)
                                                        .andArgs(Arg.DerivIdx.of(-1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                    b = reduction[ 0 ];
                } else b = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);

                alternative = goDeeperWith.execute(
                                        ExecutionCall.of( tsrs[ 0 ], a, b )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().backend().getOperation("*") )
                                                        .on( device )
                                );
                alternative = goDeeperWith.execute(
                                        ExecutionCall.of( tsrs[ 0 ], tsrs[ 0 ], tsrs[d+1] )
                                                        .andArgs(Arg.DerivIdx.of(1))
                                                        .running(Neureka.get().backend().getOperation("/"))
                                                        .on(device)
                                );
                if ( d == 0 ) a.delete();
                b.delete();
            }
            return alternative;
        } else
            return alternative;
    }

    @Contract( pure = true )
    public static Tsr<?> forAdditions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith
    ) {
        return _forAdditionsOrSubtractions(call, goDeeperWith, true);
    }

    @Contract( pure = true )
    public static Tsr<?> forSubtractions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith
    ) {
        return _forAdditionsOrSubtractions(call, goDeeperWith, false);
    }

    @Contract( pure = true )
    private static Tsr<?> _forAdditionsOrSubtractions(
            ExecutionCall<? extends Device<?>> call,
            CallExecutor goDeeperWith,
            boolean thisIsForAddition
    ) {
        Tsr<?>[] tsrs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getValOf( Arg.DerivIdx.class );
        Operation operation = call.getOperation();

        Tsr<?> alternative = null;
        if (tsrs.length > 3) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                alternative = goDeeperWith.execute(
                                    ExecutionCall.of(reduction)
                                                    .andArgs(Arg.DerivIdx.of(d))
                                                    .running(operation)
                                                    .on(device)
                            );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.execute(
                                        ExecutionCall.of(reduction)
                                                        .andArgs(Arg.DerivIdx.of(d))
                                                        .running(operation)
                                                        .on(device)
                                );
                tsrs[ 0 ] = reduction[ 0 ];
            } else {
                tsrs[ 0 ] = Tsr.Create.newTsrLike(tsrs[ 1 ]).setValue((d==0||thisIsForAddition)?1.0f:-1.0f);
            }
            return alternative;
        }
        else
            return alternative;
    }

}
