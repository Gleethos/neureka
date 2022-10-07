package neureka.backend.main.implementations.elementwise;

public class CLBiElementwiseMultiplication extends CLBiElementwise
{
    public CLBiElementwiseMultiplication(String postfix) {
        super(
            postfix,
                "output = input1 * input2;\n",
                "output = input2;\n"
        );
    }
}
