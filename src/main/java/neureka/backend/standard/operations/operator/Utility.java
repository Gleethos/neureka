package neureka.backend.standard.operations.operator;

import neureka.Tsr;
import neureka.backend.api.Algorithm;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.operations.OperationContext;
import neureka.devices.Device;

import java.util.function.Function;

public class Utility implements Algorithm.RecursiveJunctionAgent
{


    @Override
    public Tsr<?> handle(
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
                                    ExecutionCall.builder()
                                            .device(device)
                                            .tensors(reduction)
                                            .derivativeIndex(d)
                                            .operation(type)
                                            .build()
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
                                    .derivativeIndex(-1)
                                    .operation(OperationContext.get().instance("/"))
                                    .build()
                    );
                    a = reduction[ 0 ];
                } else if ( d == 1 ) a = tsrs[ 1 ];
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
                                    .derivativeIndex(-1)
                                    .operation(OperationContext.get().instance("/"))
                                    .build()
                    );
                    b = reduction[ 0 ];
                } else b = Tsr.Create.newTsrLike(tsrs[ 1 ], 1.0);

                alternative = goDeeperWith.apply(
                        ExecutionCall.builder()
                                .device(device)
                                .tensors( new Tsr[]{tsrs[ 0 ], a, b} )
                                .derivativeIndex( -1 )
                                .operation( OperationContext.get().instance("*") )
                                .build()
                );
                alternative = goDeeperWith.apply(
                        ExecutionCall.builder()
                                .device(device)
                                .tensors( new Tsr[]{tsrs[ 0 ], tsrs[ 0 ], tsrs[d+1]} )
                                .derivativeIndex( 1 )
                                .operation( OperationContext.get().instance("/") )
                                .build()
                );
                if ( d == 0 ) a.delete();
                b.delete();
            }
            return alternative;
        } else
            return alternative;
    }
}
