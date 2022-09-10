package neureka.backend.main.implementations.elementwise;

public class CLBiElementwisePower extends CLBiElementwise
{
    public CLBiElementwisePower(String postfix) {
        super(postfix,
            "output = pow(input1, input2);",
            "if ( d == 0 ) {                 \n" +
            "    output = input2 * pow(input1, input2-1.0f);  \n" +
            "} else {                                         \n" +
            "    output = pow(input1, input2) * log(input1);  \n" +
            "}"
        );
    }
}
