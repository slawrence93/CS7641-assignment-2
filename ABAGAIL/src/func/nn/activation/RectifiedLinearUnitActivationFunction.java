package func.nn.activation;

public class RectifiedLinearUnitActivationFunction extends DifferentiableActivationFunction {

	public double derivative(double value) {
		if(value < 0) {
			return 0;
		} else {
			return 1;
		}
	}

	public double value(double value) {
		if(value < 0.0) {
			return 0.0;
		} else {
			return value;
		}
	}
}