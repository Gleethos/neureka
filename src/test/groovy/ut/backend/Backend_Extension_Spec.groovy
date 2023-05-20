package ut.backend

import neureka.MutateTensor
import neureka.Neureka
import neureka.Tensor
import neureka.backend.api.Algorithm
import neureka.backend.api.BackendContext
import neureka.backend.api.Operation
import neureka.backend.api.Result
import neureka.math.implementations.FunctionInput
import neureka.math.implementations.FunctionNode
import neureka.devices.Device
import spock.lang.Specification
import spock.lang.Subject

@Subject([Operation, BackendContext])
class Backend_Extension_Spec extends Specification
{
    def setupSpec()
    {
        Neureka.get().reset()

        reportHeader """
                   This specification defines the behavior of
                   Operation instances and their ability to be extended! <br> 
        """
    }

    def 'Mock operation interacts with FunctionNode (AbstractFunction) instance as expected.'() {

        given : 'A new mock operation type is being created.'
            var op = Mock(Operation)

        and : 'A list of mocked function source nodes.'
            var children = [Mock(FunctionInput), Mock(FunctionInput)]

        and : 'A mock tensor which is the expected output'
        Tensor output = Mock(Tensor)
            var mutate = Mock(MutateTensor)

        and : 'A mocked operation implementation.'
            var implementation = Mock(Algorithm)

        when : 'A FunctionNode is being instantiated via the given mocks...'
            var function = new FunctionNode(op, children, false)

        then : 'The mock type has been called as expected and the function has the following properties.'
            (1.._) * op.getArity() >> 2
            function.isFlat()
            !function.isDoingAD()

        when : 'The function is being called with an empty tensor array...'
            var result = function.call(new Tensor[0])

        then : 'The custom call hook should be accessed as outlined below.'
            (0.._) * op.getAlgorithmFor(_) >> implementation
            (0.._) * implementation.execute(_,_) >> Result.of(output)
            (1.._) * output.getMut() >> mutate
            (1.._) * mutate.setIsIntermediate(false) >> output
            (1.._) * output.isIntermediate() >> true
            (1.._) * op.execute(_,_) >> Result.of(output)

        and : 'The mocked output tensor never returns the mock device because our custom call hook replaces execution.'
            0 * output.getDevice() >> Mock(Device)

        and : 'The ADAnalyzer of the mock implementation will not be called because "doAD" is set to "false".'
            0 * implementation.isSuitableFor(_)

        and : 'The result is the same as the mock tensor returned by the custom call hook.'
            result == output
    }

}
