package st.attention;

import neureka.Tensor;
import neureka.math.Function;
import neureka.math.args.Arg;

import java.util.List;

public class QuasiMultiHeadAttention
{
    private final Function _cat = Function.of("concat(i0, i1, i2, i3)");

    private final int _numberOfNeurons;
    private final ReductiveAttentionHead _attentionHead1;
    private final ReductiveAttentionHead _attentionHead2;
    private final ReductiveAttentionHead _attentionHead3;
    private final ReductiveAttentionHead _attentionHead4;
    private final ReductiveAttentionHead _outputHead;

    private Tensor<Float> _lastPrediction = null;

    public QuasiMultiHeadAttention(
        int numberOfNeurons,
        int numberOfInputs
    ) {
        _numberOfNeurons = numberOfNeurons;
        _attentionHead1 = new ReductiveAttentionHead(numberOfNeurons, numberOfInputs);
        _attentionHead2 = new ReductiveAttentionHead(numberOfNeurons, numberOfInputs);
        _attentionHead3 = new ReductiveAttentionHead(numberOfNeurons, numberOfInputs);
        _attentionHead4 = new ReductiveAttentionHead(numberOfNeurons, numberOfInputs);
        _outputHead     = new ReductiveAttentionHead(numberOfNeurons, 4);
    }

    public Tensor<Float> run(List<Tensor<Float>> inputs ) {
        Tensor<Float>[] inputArray = new Tensor[inputs.size()];
        for ( int i = 0; i < inputs.size(); i++ )
            inputArray[i] = inputs.get(i);

        Tensor<Float> input = inputArray.length == 1 ? inputArray[0] : _cat.with(Arg.Axis.of(0)).call(inputArray);
        Tensor<Float> headAttention1 = _attentionHead1.run(input);
        Tensor<Float> headAttention2 = _attentionHead2.run(input);
        Tensor<Float> headAttention3 = _attentionHead3.run(input);
        Tensor<Float> headAttention4 = _attentionHead4.run(input);
        Tensor<Float> input2 = _cat.with(Arg.Axis.of(0)).call(headAttention1, headAttention2, headAttention3, headAttention4);
        Tensor<Float> output = _outputHead.run(input2);
        _lastPrediction = output;
        return output;
    }


    public double train( Tensor<Float> y ) {
        Tensor<Float> error = y.minus(_lastPrediction).power(2f).sum();
        error.backward();
        _lastPrediction = null;
        _applyGradients();
        return error.item();
    }

    private void _applyGradients() {
        _attentionHead1.applyGradients();
        _attentionHead2.applyGradients();
        _attentionHead3.applyGradients();
        _attentionHead4.applyGradients();
        _outputHead.applyGradients();
    }

}
