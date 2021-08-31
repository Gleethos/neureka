package neureka.backend.standard.operations;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.calculus.args.Arg;
import neureka.devices.Device;

import java.util.function.Function;

public class JunctionUtil
{
    public static Tsr<?> forConvolution(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith
    ) {
        Tsr<?>[] tsrs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getDerivativeIndex();
        Operation operation = call.getOperation();

        Tsr<?> alternative = null;
        if (tsrs.length > 3) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ] };
                alternative = goDeeperWith.apply(
                                    ExecutionCall.of( reduction )
                                                    .andArgs( Arg.DerivIdx.of(d) )
                                                    .running( operation )
                                                    .on(device)
                                );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.apply(
                        ExecutionCall.of(reduction)
                                        .andArgs(Arg.DerivIdx.of(d))
                                        .running(operation)
                                        .on(device)
                );
                tsrs[ 0 ] = reduction[ 0 ];
            }
            return alternative;
        } else {
            if ( call.getOperation().getOperator().equals("x") ) {
                if ( d >= 0 ) {
                    if ( d == 0 ) tsrs[ 0 ] = tsrs[ 2 ];
                    else tsrs[ 0 ] = tsrs[ 1 ];
                    return tsrs[ 0 ];
                } else {
                    call.mutateArguments( t -> new Tsr[]{t[ 0 ], t[ 1 ], t[ 2 ]} );
                }
            } else if ( call.getOperation().getOperator().equals("x"+ ((char) 187)) ) {
                call.mutateArguments( t -> new Tsr[]{t[ 2 ], t[ 1 ], t[ 0 ]} );
            }
            return alternative;
        }
    }

    public static Tsr<?> forMultiplications(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith
    ) {
        Tsr<?>[] tsrs = call.getTensors();
        Device<?> device = call.getDevice();
        int d = call.getDerivativeIndex();
        Operation type = call.getOperation();

        Tsr<?> alternative = null;
        if (tsrs.length > 3) {
            if ( d < 0 ) {
                Tsr<?>[] reduction = new Tsr[]{ tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ] };
                alternative = goDeeperWith.apply(
                        ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.apply(
                        ExecutionCall.of(reduction).andArgs(Arg.DerivIdx.of(d)).running(type).on(device)
                );
                tsrs[ 0 ] = reduction[ 0 ];
            } else {
                Tsr[] reduction = Operation.Utility.without(tsrs, 1+d);
                if ( reduction.length > 2 ) {
                    reduction[ 0 ] = ( reduction[ 0 ] == null ) ? Tsr.Create.newTsrLike(tsrs[ 1 ]) : reduction[ 0 ];
                    alternative = goDeeperWith.apply(
                            ExecutionCall.of(reduction)
                                            .andArgs( Arg.DerivIdx.of( -1 ) )
                                            .running( Neureka.get().context().instance("*") )
                                            .on( device )
                    );
                    tsrs[ 0 ] = reduction[ 0 ];
                } else tsrs[ 0 ] = reduction[ 1 ];
            }
            return alternative;
        } else
            return alternative;

    }


    public static Tsr<?> forDivisionsOrModuli(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith
    ) {
        Tsr[] tsrs = call.getTensors();
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        Operation type = call.getOperation();

        Tsr alternative = null;
        if ( tsrs.length > 3 )
        {
            if ( d < 0 ) {
                Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                alternative = goDeeperWith.apply(
                        call.withTensors( reduction )
                            );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.apply(
                                    call.withTensors(reduction)
                            );
                tsrs[ 0 ] = reduction[ 0 ];
            } else {
                Tsr a;
                if ( d > 1 ) {
                    Tsr[] reduction = Operation.Utility.subset(tsrs, 1, 1, d+1);
                    reduction[ 0 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ]);
                    alternative = goDeeperWith.apply(
                                        ExecutionCall.builder()
                                                .device(device)
                                                .tensors(reduction)
                                                .operation(Neureka.get().context().instance("/"))
                                                .args(
                                                        Arg.DerivIdx.of(-1)
                                                )
                                                .build()
                    );
                    a = reduction[ 0 ];
                }
                else if ( d == 1 ) a = tsrs[ 1 ];
                else a = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                Tsr b;
                if ( tsrs.length -  d - 2  > 1 ) {
                    Tsr[] reduction = Operation.Utility.subset(tsrs, 2, d+2, tsrs.length-(d+2));
                    reduction[ 1 ] =  Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);
                    reduction[ 0 ] = reduction[ 1 ];
                    alternative = goDeeperWith.apply(
                            ExecutionCall.builder()
                                    .device(device)
                                    .tensors(reduction)
                                    .operation(Neureka.get().context().instance("/"))
                                    .args(
                                            Arg.DerivIdx.of(-1)
                                    )
                                    .build()
                    );
                    b = reduction[ 0 ];
                } else b = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);

                alternative = goDeeperWith.apply(
                                        ExecutionCall.of( tsrs[ 0 ], a, b )
                                                        .andArgs( Arg.DerivIdx.of( -1 ) )
                                                        .running( Neureka.get().context().instance("*") )
                                                        .on( device )
                                );
                alternative = goDeeperWith.apply(
                                        ExecutionCall.of( tsrs[ 0 ], tsrs[ 0 ], tsrs[d+1] )
                                                        .andArgs(Arg.DerivIdx.of(1))
                                                        .running(Neureka.get().context().instance("/"))
                                                        .on(device)
                                );
                if ( d == 0 ) a.delete();
                b.delete();
            }
            return alternative;
        } else
            return alternative;
    }

    public static Tsr<?> forAdditions(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith
    ) {
        return _forAdditionsOrSubtractions(call, goDeeperWith, true);
    }

    public static Tsr<?> forSubtractions(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith
    ) {
        return _forAdditionsOrSubtractions(call, goDeeperWith, false);
    }

    private static Tsr<?> _forAdditionsOrSubtractions(
            ExecutionCall<? extends Device<?>> call,
            Function<ExecutionCall<? extends Device<?>>, Tsr<?>> goDeeperWith,
            boolean thisIsForAddition
    ) {
        Tsr[] tsrs = call.getTensors();
        Device device = call.getDevice();
        int d = call.getDerivativeIndex();
        Operation operation = call.getOperation();

        Tsr alternative = null;
        if (tsrs.length > 3) {
            if ( d < 0 ) {
                Tsr[] reduction = new Tsr[]{tsrs[ 0 ], tsrs[ 1 ], tsrs[ 2 ]};
                alternative = goDeeperWith.apply(
                        ExecutionCall.builder()
                                .device(device)
                                .tensors(reduction)
                                .operation(operation)
                                .args(
                                        Arg.DerivIdx.of(d)
                                )
                                .build()
                );
                tsrs[ 0 ] = reduction[ 0 ];

                reduction = Operation.Utility.offsetted(tsrs, 1);
                alternative = goDeeperWith.apply(
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
        } else
            return alternative;
    }

}
