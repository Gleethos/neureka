package neureka.backend.main.implementations.elementwise;

public class CLBiElementwiseDivision extends CLBiElementwise
{
    public CLBiElementwiseDivision(String postfix) {
        super(
            postfix,
                "output = input1 / input2;\n",
                "output = ( d == 0 ? 1 / input2 : -input2 / (float)pow(input1, 2.0f)  );  \n"
        );
    }
}
