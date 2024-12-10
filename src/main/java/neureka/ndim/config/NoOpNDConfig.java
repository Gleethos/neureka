package neureka.ndim.config;

final class NoOpNDConfig implements NDConfiguration
{

    static final NoOpNDConfig INSTANCE = new NoOpNDConfig();

    private NoOpNDConfig() {}

    @Override
    public int rank() {
        return 0;
    }

    @Override
    public int[] shape() {
        return new int[0];
    }

    @Override
    public int shape(int i) {
        return 0;
    }

    @Override
    public int[] indicesMap() {
        return new int[0];
    }

    @Override
    public int indicesMap(int i) {
        return 0;
    }

    @Override
    public int[] strides() {
        return new int[0];
    }

    @Override
    public int strides(int i) {
        return 0;
    }

    @Override
    public int[] spread() {
        return new int[0];
    }

    @Override
    public int spread(int i) {
        return 0;
    }

    @Override
    public int[] offset() {
        return new int[0];
    }

    @Override
    public int offset(int i) {
        return 0;
    }

    @Override
    public int indexOfIndex(int index) {
        return 0;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[0];
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return 0;
    }

    @Override
    public boolean equals(NDConfiguration ndc) {
        return false;
    }

    @Override
    public NDConfiguration newReshaped(int[] newForm) {
        return this;
    }
}
