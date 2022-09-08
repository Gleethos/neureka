package neureka.backend.main.implementations.broadcast;

public class CLBroadcastAddition extends CLBroadcast
{
    public CLBroadcastAddition(String id) {
        super(id, "value += src1 + src2;\n", "value += 1 * drain;\n");
    }
}
