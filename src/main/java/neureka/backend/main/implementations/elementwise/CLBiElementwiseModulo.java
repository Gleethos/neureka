package neureka.backend.main.implementations.elementwise;

public class CLBiElementwiseModulo extends CLBiElementwise
{
    public CLBiElementwiseModulo(String postfix) {
        super(
            postfix,
                "output = ((int)input1) % ((int)input2);\n",
                "output = ( d == 0 ? 1/input2 : -input2 / (float) pow(input1, 2.0f) );\n"
        );
    }
}
