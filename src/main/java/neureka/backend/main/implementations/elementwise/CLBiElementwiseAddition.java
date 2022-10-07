package neureka.backend.main.implementations.elementwise;

public class CLBiElementwiseAddition extends CLBiElementwise
{
    public CLBiElementwiseAddition(String postfix) {
        super(
            postfix,
             "output = input1 + input2;\n",
             "output = 1;\n"
        );
    }
}
