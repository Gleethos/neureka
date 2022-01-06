package ut.backend

import neureka.Neureka
import neureka.Tsr
import neureka.devices.host.CPU
import neureka.devices.opencl.KernelCaller
import neureka.devices.opencl.OpenCLDevice
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Algorithm
import neureka.backend.standard.algorithms.Activation
import neureka.backend.standard.algorithms.Operator
import spock.lang.Specification

class Backend_Algorithm_Implementation_Spec extends Specification
{
    def setupSpec()
    {
        Neureka.get().reset()

        reportHeader """
                   This specification defines the behavior of implementations of the 
                   Algorithm interface! <br> 
        """
    }

    def 'Operator implementations have expected Executor instances.'(
            Algorithm imp
    ){

        when : 'Host- and CL- executor instance are being fetched...'
            def hostExecutor = imp.getImplementationFor( CPU.class )
            def clExecutor = imp.getImplementationFor( OpenCLDevice.class )

        then : 'The variables containing the executor instances are not null.'
            hostExecutor != null
            clExecutor != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << Neureka.get().backend()
                    .getOperations()
                    .stream()
                    .filter(
                        e ->
                                    e.isOperator() &&
                                    e.getOperator().length() == 1 &&
                                    e.supports( Operator.class )
                    ).map( e -> e.getAlgorithm( Operator.class ) )

    }



    def 'Activation implementations have expected Executor instances.'(
            Algorithm imp
    ){

        when : 'Host- and CL- executor instance are being fetched...'
            def hostExecutor = imp.getImplementationFor( CPU.class )
            def clExecutor = imp.getImplementationFor( OpenCLDevice.class )

        then : 'The variables containing the executor instances are not null.'
        hostExecutor != null
        clExecutor != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
        imp << Neureka.get().backend()
                .getOperations()
                .stream()
                .filter(
                        e ->
                                        e.supports( Activation.class )
                ).map( e -> e.getAlgorithm( Activation.class ) )

    }


    def 'HostExecutors of Operator implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'Mock instances to simulate an ExecutionCall instance.'
            def call = Mock( ExecutionCall )
            def device = Mock( CPU )
            def tensor = Mock( Tsr )
            def hostExecutor = imp.getImplementationFor( CPU.class )
            def nativeExecutor = Mock( CPU.JVMExecutor )

        when : 'Host-executor instance is being called...'
            hostExecutor.run( call )

        then : 'The mock objects are being called as expected.'
            (1.._) * call.getDevice() >> device
            1 * device.getExecutor() >> nativeExecutor
            1 * nativeExecutor.threaded( _, _ )
            (1.._) * call.getTsrOfType( Number.class, 0) >> tensor
            (1.._) * tensor.size() >> 0

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << Neureka.get().backend()
                            .getOperations()
                            .stream()
                            .filter(
                                    e ->
                                            e.isOperator() &&
                                                    e.getOperator().length() == 1 &&
                                                    e.supports( Operator.class )
                            )
                            .map( e -> e.getAlgorithm( Operator.class ) )

    }



    def 'CLExecutors of Operator implementations behave as expected.'(
            Algorithm imp
    ){

        given : 'Mock instances to simulate an ExecutionCall instance.'
            def call = Mock( ExecutionCall )
            def device = Mock( OpenCLDevice )
            def tensor = Mock( Tsr )
            def clExecutor = imp.getImplementationFor( OpenCLDevice.class )
            def kernel = Mock( KernelCaller )

        when : 'CL-executor instance is being called...'
            clExecutor.run( call )

        then : 'The mock objects are being called as expected.'
            (1.._) * call.getTsrOfType( Number.class, 0) >> tensor
            (1.._) * tensor.size() >> 0
            (1.._) * call.getDevice() >> device
             1 * device.getKernel(call) >> kernel
            (1.._) * kernel.passAllOf(_) >> kernel
            (1.._) * kernel.pass(_) >> kernel
            (1.._) * kernel.call(_)

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            imp << Neureka.get().backend()
                            .getOperations()
                            .stream()
                            .filter(
                                    e ->
                                            e.isOperator() &&
                                            e.getOperator().length() == 1 &&
                                            e.supports( Operator.class )
                            )
                            .map( e -> e.getAlgorithm( Operator.class ) )

    }



}