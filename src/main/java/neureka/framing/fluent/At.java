package neureka.framing.fluent;

/**
 *
 * @param <K> The key type which will be provided by the user of this method.
 * @param <R> The return type which will be provided by an implementation of this method.
 */
public interface At<K, R> {

    R at(K key );

}
