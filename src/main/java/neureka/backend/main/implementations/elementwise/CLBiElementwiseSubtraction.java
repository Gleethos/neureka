package neureka.backend.main.implementations.elementwise;

public class CLBiElementwiseSubtraction extends CLBiElementwise
{
    public CLBiElementwiseSubtraction(String postfix) {
        super(
            postfix,
                "output = input1 - input2;\n",
                "output = 1;\n"
        );
    }
}
