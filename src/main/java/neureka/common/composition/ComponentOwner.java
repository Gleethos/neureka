package neureka.common.composition;

import java.util.List;
import java.util.function.Consumer;

public interface ComponentOwner<C> {
    <T extends Component<?>> T get(Class<T> componentClass);

    <T extends Component<?>> List<T> getAll(Class<T> componentClass);

    <T extends Component<C>> C remove(Class<T> componentClass);

    <T extends Component<C>> boolean has(Class<T> componentClass);

    <T extends Component<C>> C set(T newComponent);

    <T extends Component<C>> boolean forComponent(Class<T> cc, Consumer<T> action);
}
