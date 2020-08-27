package ut.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.acceleration.host.HostCPU
import neureka.acceleration.host.execution.HostExecutor
import neureka.acceleration.opencl.KernelBuilder
import neureka.acceleration.opencl.OpenCLDevice
import neureka.acceleration.opencl.execution.CLExecutor
import neureka.calculus.environment.ExecutionCall
import neureka.calculus.environment.OperationTypeImplementation
import neureka.calculus.environment.implementations.Activation
import neureka.calculus.environment.implementations.Operator
import neureka.calculus.environment.operations.OperationContext
import spock.lang.Specification

class Calculus_Implementation_Executor_Tests extends Specification
{

    def 'Operator implementations have expected Executor instances.'(
            OperationTypeImplementation imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        when : 'Host- and CL- executor instance are being fetched...'
            def hostExecutor = imp.getExecutor( HostExecutor.class )
            def clExecutor = imp.getExecutor( CLExecutor.class )

        then : 'The variables containing the executor instances are not null.'
            hostExecutor != null
            clExecutor != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << OperationContext.instance()
                    .instances()
                    .stream()
                    .filter(
                        e ->
                                    e.isOperator() &&
                                    e.getOperator().length() == 1 &&
                                    e.supports( Operator.class )
                    ).map( e -> e.getImplementation( Operator.class ) )

    }



    def 'Activation implementations have expected Executor instances.'(
            OperationTypeImplementation imp
    ){

        given : 'The current Neureka instance is being reset.'
        Neureka.instance().reset()

        when : 'Host- and CL- executor instance are being fetched...'
        def hostExecutor = imp.getExecutor( HostExecutor.class )
        def clExecutor = imp.getExecutor( CLExecutor.class )

        then : 'The variables containing the executor instances are not null.'
        hostExecutor != null
        clExecutor != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
        imp << OperationContext.instance()
                .instances()
                .stream()
                .filter(
                        e ->
                                        e.supports( Activation.class )
                ).map( e -> e.getImplementation( Activation.class ) )

    }


    def 'HostExecutors of Operator implementations behave as expected.'(
            OperationTypeImplementation imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'Mock instances to simulate an ExecutionCall instance.'
            def call = Mock( ExecutionCall )
            def device = Mock( HostCPU )
            def tensor = Mock( Tsr )
            def hostExecutor = imp.getExecutor( HostExecutor.class )
            def nativeExecutor = Mock( HostCPU.NativeExecutor )

        when : 'Host-executor instance is being called...'
            hostExecutor.execution.call( call )

        then : 'The mock objects are being called as expected.'
            (1.._) * call.getDevice() >> device
            1 * device.getExecutor() >> nativeExecutor
            1 * nativeExecutor.threaded( _, _ )
            (1.._) * call.getTensor(0) >> tensor
            (1.._) * tensor.size() >> 0

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << OperationContext.instance()
                .instances()
                .stream()
                .filter(
                        e ->
                                e.isOperator() &&
                                        e.getOperator().length() == 1 &&
                                        e.supports( Operator.class )
                ).map( e -> e.getImplementation( Operator.class ) )

    }



    def 'CLExecutors of Operator implementations behave as expected.'(
            OperationTypeImplementation imp
    ){

        given : 'The current Neureka instance is being reset.'
            Neureka.instance().reset()

        and : 'Mock instances to simulate an ExecutionCall instance.'
            def call = Mock( ExecutionCall )
            def device = Mock( OpenCLDevice )
            def tensor = Mock( Tsr )
            def clExecutor = imp.getExecutor( CLExecutor.class )
            def kernel = Mock( KernelBuilder )

        when : 'CL-executor instance is being called...'
            clExecutor.execution.call( call )

        then : 'The mock objects are being called as expected.'
            (1.._) * call.getTensor(0) >> tensor
            (1.._) * tensor.size() >> 0
            (1.._) * call.getDevice() >> device
             1 * device.getKernel(call) >> kernel
            (1.._) * kernel.pass(_) >> kernel
            (1.._) * kernel.call(_)

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << OperationContext.instance()
                .instances()
                .stream()
                .filter(
                        e ->
                                e.isOperator() &&
                                        e.getOperator().length() == 1 &&
                                        e.supports( Operator.class )
                ).map( e -> e.getImplementation( Operator.class ) )

    }



}
