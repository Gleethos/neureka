package st.attention;

import neureka.Neureka;
import neureka.Tsr;
import neureka.math.Function;
import neureka.optimization.Optimizer;

public class ReductiveAttentionHead
{
    private final Function _f  = Neureka.get().backend().getAutogradFunction().tanh();


    private final int _numberOfNeurons;
    private final Tsr<Float> _keyWeights;
    private final Tsr<Float> _queryWeights;
    private final Tsr<Float> _valueWeights;

    private final Tsr<Float> _forwardWeights1;
    private final Tsr<Float> _forwardWeights2;
    private final Tsr<Float> _forwardBias1;
    private final Tsr<Float> _forwardBias2;

    private final double[] _attentionScores;

    public ReductiveAttentionHead(
        int numberOfNeurons,
        int numberOfInputs
    ) {
        _numberOfNeurons = numberOfNeurons;
        int flattened = numberOfNeurons * numberOfInputs;
        int seed = flattened + numberOfNeurons + numberOfInputs;
        _keyWeights      = Tsr.ofFloats().withShape(numberOfNeurons, numberOfNeurons).andSeed(seed + 1).setRqsGradient(true);
        _queryWeights    = Tsr.ofFloats().withShape(numberOfNeurons, numberOfNeurons).andSeed(seed + 2).setRqsGradient(true);
        _valueWeights    = Tsr.ofFloats().withShape(numberOfNeurons, numberOfNeurons).andSeed(seed + 3).setRqsGradient(true);
        _forwardWeights1 = Tsr.ofFloats().withShape(numberOfNeurons, numberOfNeurons).andSeed(seed + 4).setRqsGradient(true);
        _forwardWeights2 = Tsr.ofFloats().withShape(flattened, numberOfNeurons).andSeed(seed + 5).setRqsGradient(true);
        _forwardBias1    = Tsr.ofFloats().withShape(1, numberOfNeurons).andSeed(seed + 6).setRqsGradient(true);
        _forwardBias2    = Tsr.ofFloats().withShape(1, numberOfNeurons).andSeed(seed + 7).setRqsGradient(true);

        _keyWeights.set(Optimizer.ADAM);
        _queryWeights.set(Optimizer.ADAM);
        _valueWeights.set(Optimizer.ADAM);
        _forwardWeights1.set(Optimizer.ADAM);
        _forwardWeights2.set(Optimizer.ADAM);
        _forwardBias1.set(Optimizer.ADAM);
        _forwardBias2.set(Optimizer.ADAM);

        _attentionScores = new double[numberOfInputs];
    }

    public Tsr<Float> run( Tsr<Float> input ) {
        Tsr<Float> key       = input.matMul(_keyWeights);
        Tsr<Float> query     = input.matMul(_queryWeights);
        Tsr<Float> value     = input.matMul(_valueWeights); // (numberOfInputs, numberOfNeurons)
        key.mut().label("key");
        query.mut().label("query");
        value.mut().label("value");
        Tsr<Float> attention = query.matMul(key.T()).div( (float) Math.sqrt(_numberOfNeurons) ).softmax(1);

        Tsr<Float> attentionSum = attention.detached().sum(0);
        for ( int i = 0; i < _attentionScores.length; i++ )
            _attentionScores[i] = attentionSum.item(i);

        /*
            Here we now have a matrix where the rows sum up to 1.
            They represent the attention weights for each input.
        */
        Tsr<Float> output    = attention.matMul(value);
        Tsr<Float> forward1  = _f.call(output.matMul(_forwardWeights1).plus(_forwardBias1));
        forward1 = forward1.reshape(1, forward1.shape(0) * forward1.shape(1));
        Tsr<Float> forward2  = _f.call(forward1.matMul(_forwardWeights2).plus(_forwardBias2));
        return forward2;
    }

    public void applyGradients() {
        _keyWeights.applyGradient();
        _queryWeights.applyGradient();
        _valueWeights.applyGradient();
        _forwardWeights1.applyGradient();
        _forwardWeights2.applyGradient();
        _forwardBias1.applyGradient();
        _forwardBias2.applyGradient();
    }

}
