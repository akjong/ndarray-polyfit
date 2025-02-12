use ndarray::Array1;
use ndarray::Array2;
use ndarray_linalg::Solve;

/// @param x_values The x-values
/// @param y_values The y-values
/// @param polynomial_degree The degree of the polynomial. I. e. 2 for a parabola.
/// @return Degree of monomials increases with the vector index
pub fn polyfit(
    x_values: Array1<f64>,
    y_values: Array1<f64>,
    polynomial_degree: usize,
) -> Result<Array1<f64>, &'static str> {
    let number_of_columns = polynomial_degree + 1;
    let number_of_rows = x_values.len();

    // Construct the Vandermonde matrix
    let mut a = Array2::zeros((number_of_rows, number_of_columns));

    for row in 0..number_of_rows {
        // First column is always 1
        a[[row, 0]] = 1.0;

        for col in 1..number_of_columns {
            a[[row, col]] = x_values[row].powf(col as f64);
        }
    }

    // Solve using least squares method
    match a.solve(&y_values) {
        Ok(mat) => Ok(mat),
        Err(_error) => Err("Linear system could not be solved"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_polyfit() {
        let x_values = array![0.0, 1.0, 2.0];
        let y_values = array![0.0, 1.0, 4.0];

        let result = polyfit(x_values, y_values, 2).unwrap();

        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-6);
    }
}
