package ut.backend.core


import neureka.Neureka
import neureka.backend.api.Algorithm
import neureka.backend.api.AutoDiffMode
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.main.algorithms.Activation
import neureka.backend.main.algorithms.BiElementWise
import neureka.backend.main.algorithms.Broadcast
import neureka.backend.main.algorithms.NDConvolution
import neureka.devices.Device
import spock.lang.Specification

class Backend_Algorithm_AD_Spec extends Specification
{
    def 'Operator implementations behave as expected.'( Algorithm alg )
    {
        given : 'The current Neureka instance is being reset.'
            Neureka.get().reset()

        and : 'A mock ExecutionCall.'
            var call = ExecutionCall.of().running(Mock(Operation)).on(Mock(Device))

        expect : 'The algorithm is not null.'
            alg != null
        and : 'The algorithm is a bi-elementwise algorithm.'
            alg instanceof BiElementWise
        and : 'It has a non empty name and string representation.'
            !alg.name.isEmpty()
            !alg.toString().isEmpty()

        when : 'A auto diff mode is being created by every algorithm...'
            AutoDiffMode mode = alg.autoDiffModeFrom( call )

        then : 'The agent is configured to perform forward-AD and it contains the derivative generated by the function!'
            mode != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Operator".'
            alg << Neureka.get().backend()
                    .getOperations()
                    .stream()
                    .filter( e -> e.isOperator() && e.getOperator().length()==1 && e.supports( BiElementWise.class ) )
                    .map( e -> e.getAlgorithm( BiElementWise.class ) )
    }


    def 'Activation implementations behave as expected.'( Algorithm alg ){

        given : 'The current Neureka instance is being reset.'
            Neureka.get().reset()
        and : 'A mock ExecutionCall.'
            var call = ExecutionCall.of().running(Mock(Operation)).on(Mock(Device))

        expect : 'The algorithm is not null.'
            alg != null
        and : 'The algorithm is a activation algorithm.'
            alg instanceof Activation
        and : 'It has a non empty name and string representation.'
            !alg.name.isEmpty()
            !alg.toString().isEmpty()

        when :
            var suitability = alg.isSuitableFor(call)
        then :
            0 <= suitability && suitability <= 1

        when :
            var mode = alg.autoDiffModeFrom(call)
        then :
            mode != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Activation".'
            alg << Neureka.get().backend()
                    .getOperations()
                    .stream()
                    .filter( e -> e.supports( Activation.class ) )
                    .map( e -> e.getAlgorithm( Activation.class ) )
    }

    def 'Convolution implementations behave as expected.'( Algorithm alg )
    {
        given : 'The current Neureka instance is being reset.'
            Neureka.get().reset()
        and : 'A mock ExecutionCall.'
            var call = ExecutionCall.of().running(Mock(Operation)).on(Mock(Device))

        expect : 'The algorithm is not null.'
            alg != null
        and : 'The algorithm is a convolution algorithm.'
            alg instanceof NDConvolution
        and : 'It has a non empty name and string representation.'
            !alg.name.isEmpty()
            !alg.toString().isEmpty()

        when :
            var suitability = alg.isSuitableFor(call)
        then :
            0 <= suitability && suitability <= 1

        when :
            var mode = alg.autoDiffModeFrom(call)
        then :
            mode != null

        where : 'The variable "imp" is from a List of Operation implementations of type "Convolution".'
            alg << Neureka.get().backend()
                                .getOperations()
                                .stream()
                                .filter(
                                        e ->
                                                e.isOperator() &&
                                                        e.getOperator().length()==1 &&
                                                            e.supports( NDConvolution.class )
                                ).map( e -> e.getAlgorithm( NDConvolution.class ) )
    }


    def 'Broadcast implementations have expected properties.'( Algorithm alg )
    {
        given : 'We first reset the library settings.'
            Neureka.get().reset()
        and : 'A mock ExecutionCall.'
            var call = ExecutionCall.of().running(Mock(Operation)).on(Mock(Device))

        expect : 'The algorithm is not null.'
            alg != null
        and : 'The algorithm is a broadcast algorithm.'
            alg instanceof Broadcast
        and : 'It has a non empty name and string representation.'
            !alg.name.isEmpty()
            !alg.toString().isEmpty()

        when :
            var suitability = alg.isSuitableFor(call)
        then :
            0 <= suitability && suitability <= 1

        when :
            var mode = alg.autoDiffModeFrom(call)
        then :
            mode != null

        where : 'The variable "imp" is from a List of OperationType implementations of type "Convolution".'
            alg << Neureka.get().backend()
                                .getOperations()
                                .stream()
                                .filter( e -> e.isOperator() && e.supports( Broadcast.class ) )
                                .map( e -> e.getAlgorithm( Broadcast.class ) )
    }

}
