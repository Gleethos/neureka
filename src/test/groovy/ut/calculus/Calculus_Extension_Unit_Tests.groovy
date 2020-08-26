package ut.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.Device
import neureka.autograd.ADAgent
import neureka.calculus.Function
import neureka.calculus.environment.ExecutionCall
import neureka.calculus.environment.OperationType
import neureka.calculus.environment.implementations.GenericImplementation
import neureka.calculus.environment.implementations.Operator
import neureka.calculus.environment.operations.OperationContext
import neureka.calculus.factory.assembly.FunctionBuilder
import spock.lang.Specification

class Calculus_Extension_Unit_Tests extends Specification
{

   /*     // TODO
    def 'GEMM matrix multiplication reference implementation can be set as custom OperationType.'(){



    }
    */


    //def '...'(){
//
    //    given : 'Neureka is being reset.'
    //        Neureka.instance().reset()
//
    //    and : 'A new OperationContext for testing.'
    //        OperationContext oldContext = OperationContext.instance()
    //        OperationContext context = OperationContext.instance().clone()
    //        OperationContext.setInstance(context)
//
    //    and : 'A new operation type with a new implementation.'
    //        new OperationType(
    //                "test_operation", "o", 2,
    //                false, false, false, false, false
    //        ).setImplementation(
    //                GenericImplementation.class,
    //                new GenericImplementation()
    //                        .setHandleChecker(call -> true)
    //                        .setADAnalyzer(call -> false)
    //                        .setADAgentCreator(
    //                            (Function f, ExecutionCall<Device> call, boolean forward ) ->
    //                            {
    //                                Tsr derivv = (Tsr)call.getAt("derivative");
    //                                return new ADAgent(
    //                                        null
    //                                ).withForward(
    //                                        (t, derivative) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{derivative},0)
    //                                ).withBackward(
    //                                        (t, error) -> FunctionBuilder.build(f.toString(), false).derive(new Tsr[]{error},0)
    //                                );
    //                            }
    //                        ).setCallHock(
    //                            ( caller, call ) -> { return null; }
    //                        ).setRJAgent(
    //                            ( call, goDeeperWith ) -> { return null; }
    //                        ).setDrainInstantiation(
    //                            call ->
    //                            {
    //                                Tsr[] tsrs = call.getTensors();
    //                                Device device = call.getDevice();
    //                                if ( tsrs[0] == null ) // Creating a new tensor:
    //                                {
    //                                    int[] shp = tsrs[1].getNDConf().shape();
    //                                    Tsr output = new Tsr( shp, 0.0 );
    //                                    output.setIsVirtual( false );
    //                                    device.add(output);
    //                                    tsrs[0] = output;
    //                                }
    //                                return call;
    //                            }
    //                        )
    //        )
    //
    //
    //
    //}


}
