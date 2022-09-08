package neureka.backend.main.implementations.broadcast;

public class CLBroadcastMultiplication extends CLBroadcast
{
    public CLBroadcastMultiplication(String id) {
        super(id, "value = src1 * src2;\n", "value += ( d == 0 ? drain : handle );\n");
    }
}
