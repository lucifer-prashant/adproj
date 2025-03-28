import { useState, useEffect } from "react"
import {
	Container,
	Form,
	Button,
	Row,
	Col,
	Spinner,
	Card,
	Alert,
} from "react-bootstrap"
import axios from "axios"
import "bootstrap/dist/css/bootstrap.min.css"

const CircularProgress = ({
	value,
	size = 120,
	strokeWidth = 10,
	color = "#17a2b8",
}) => {
	const radius = (size - strokeWidth) / 2
	const circumference = radius * 2 * Math.PI
	const progress = value * circumference

	return (
		<div
			style={{
				position: "relative",
				width: size,
				height: size,
				margin: "0 auto",
			}}>
			<svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
				<circle
					stroke="#343a40"
					strokeWidth={strokeWidth}
					fill="transparent"
					r={radius}
					cx={size / 2}
					cy={size / 2}
				/>
				<circle
					stroke={color}
					strokeWidth={strokeWidth}
					strokeDasharray={circumference}
					strokeDashoffset={circumference - progress}
					fill="transparent"
					r={radius}
					cx={size / 2}
					cy={size / 2}
					style={{ transition: "stroke-dashoffset 0.5s ease" }}
				/>
			</svg>
			<div
				style={{
					position: "absolute",
					top: "50%",
					left: "50%",
					transform: "translate(-50%, -50%)",
					fontSize: "1.25rem",
					fontWeight: "bold",
					color: "#f8f9fa",
				}}>
				{(value * 100).toFixed(1)}%
			</div>
		</div>
	)
}

function App() {
	const [formData, setFormData] = useState({
		age: "50",
		sex: "1",
		bmi: "25.3",
		bp: "0.03",
		s1: "0.02",
		s2: "0.01",
		s3: "-0.02",
		s4: "0.01",
		s5: "0.03",
		s6: "-0.01",
	})

	const [modelType, setModelType] = useState("")
	const [result, setResult] = useState(null)
	const [availableModels, setAvailableModels] = useState([])
	const [loadingModels, setLoadingModels] = useState(true)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState(null)

	const fieldLabels = {
		age: "Age",
		sex: "Gender (0 = Female, 1 = Male)",
		bmi: "Body Mass Index (BMI)",
		bp: "Blood Pressure Level",
		s1: "Total Cholesterol",
		s2: "LDL Cholesterol",
		s3: "HDL Cholesterol",
		s4: "Triglycerides",
		s5: "Serum Glucose Level",
		s6: "Liver Enzyme Level",
	}

	useEffect(() => {
		const fetchModels = async () => {
			try {
				const response = await axios.get("http://127.0.0.1:8000/models")
				setAvailableModels(response.data.models)
				setModelType(response.data.models[0]?.name || "")
			} catch (error) {
				setError("Failed to fetch available models")
			} finally {
				setLoadingModels(false)
			}
		}
		fetchModels()
	}, [])

	const handleInputChange = (e) => {
		const { name, value } = e.target
		setFormData((prev) => ({
			...prev,
			[name]: value,
		}))
	}

	const handleSubmit = async (e) => {
		e.preventDefault()
		setLoading(true)
		setError(null)
		try {
			const response = await axios.post("http://127.0.0.1:8000/predict", {
				model_type: modelType,
				input_data: formData,
			})
			setResult(response.data)
		} catch (error) {
			setError("Failed to get prediction")
		} finally {
			setLoading(false)
		}
	}

	return (
		<div
			style={{
				backgroundColor: "#121416",
				minHeight: "100vh",
				padding: "2rem",
				margin: 0,
				fontFamily: "'Inter', sans-serif",
			}}>
			<Container fluid>
				<h1 className="text-center mb-4 text-light">Diabetes Prediction</h1>
				<Row className="justify-content-center">
					<Col md={10}>
						<Form onSubmit={handleSubmit}>
							<Card
								className="mb-4"
								style={{
									backgroundColor: "#1E2326",
									border: "1px solid #2C3236",
									borderRadius: "10px",
									boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
								}}>
								<Card.Header
									className="text-light"
									style={{
										backgroundColor: "#2C3236",
										borderBottom: "1px solid #3A4044",
									}}>
									Model Selection
								</Card.Header>
								<Card.Body>
									<Form.Group>
										{loadingModels ? (
											<div className="d-flex justify-content-center">
												<Spinner animation="border" variant="light" />
											</div>
										) : (
											<div className="d-flex flex-column">
												{availableModels.map((model) => (
													<Button
														key={model.name}
														variant={
															modelType === model.name
																? "info"
																: "outline-secondary"
														}
														onClick={() => setModelType(model.name)}
														className="mb-2"
														style={{
															backgroundColor:
																modelType === model.name
																	? "#17a2b8"
																	: "transparent",
															color:
																modelType === model.name ? "white" : "#6c757d",
															display: "flex",
															justifyContent: "space-between",
															alignItems: "center",
														}}>
														<span>{model.description}</span>
														<span className="badge bg-dark">
															{(model.accuracy * 100).toFixed(2)}%
														</span>
													</Button>
												))}
											</div>
										)}
									</Form.Group>
								</Card.Body>
							</Card>

							{/* Rest of the code remains the same as in the previous artifact */}
							<Card
								className="mb-4"
								style={{
									backgroundColor: "#1E2326",
									border: "1px solid #2C3236",
									borderRadius: "10px",
									boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
								}}>
								<Card.Header
									className="text-light"
									style={{
										backgroundColor: "#2C3236",
										borderBottom: "1px solid #3A4044",
									}}>
									Input Parameters
								</Card.Header>
								<Card.Body>
									<Row>
										{Object.keys(formData).map((field) => (
											<Col md={4} key={field} className="mb-3">
												<Form.Group controlId={field}>
													<Form.Label className="text-light">
														{fieldLabels[field]}
													</Form.Label>
													<Form.Control
														type="number"
														step="0.01"
														name={field}
														value={formData[field]}
														onChange={handleInputChange}
														placeholder={`Enter ${fieldLabels[field]}`}
														style={{
															backgroundColor: "#2C3236",
															borderColor: "#3A4044",
															color: "#f8f9fa",
														}}
													/>
												</Form.Group>
											</Col>
										))}
									</Row>
								</Card.Body>
							</Card>

							{/* Remaining code stays the same as in the previous artifact */}
							<Button
								variant="info"
								type="submit"
								disabled={loading}
								className="w-100 mb-3"
								style={{
									backgroundColor: "#17a2b8",
									borderColor: "#17a2b8",
								}}>
								{loading ? (
									<>
										<Spinner
											as="span"
											animation="border"
											size="sm"
											role="status"
											aria-hidden="true"
											className="me-2"
										/>
										Predicting...
									</>
								) : (
									"Predict"
								)}
							</Button>

							{error && (
								<Alert
									variant="danger"
									className="mt-3"
									style={{
										backgroundColor: "#721c24",
										color: "#f8d7da",
										border: "1px solid #f5c6cb",
									}}>
									{error}
								</Alert>
							)}

							{result && (
								<Card
									className="mt-4"
									style={{
										backgroundColor: "#1E2326",
										border: "1px solid #2C3236",
										borderRadius: "10px",
										boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
									}}>
									<Card.Header
										className="text-light"
										style={{
											backgroundColor: "#2C3236",
											borderBottom: "1px solid #3A4044",
										}}>
										Prediction Report
									</Card.Header>
									<Card.Body>
										<Row className="align-items-center">
											<Col md={4} className="text-center mb-4">
												<h5 className="text-light mb-3">Model Accuracy</h5>
												<CircularProgress
													value={result.model_accuracy || 0}
													color="#17a2b8"
												/>
											</Col>
											<Col md={4} className="text-center mb-4">
												<h5 className="text-light mb-3">Diabetes Risk</h5>
												<div className="d-flex flex-column align-items-center">
													<div
														className={`p-3 rounded-circle mb-2 shadow ${result.prediction === 1 ? "bg-danger" : "bg-success"}`}
														style={{
															width: "100px",
															height: "100px",
															display: "flex",
															alignItems: "center",
															justifyContent: "center",
															transition: "all 0.3s ease",
														}}>
														<span className="text-white font-weight-bold h5 mb-0">
															{result.prediction === 1 ? "High" : "Low"}
														</span>
													</div>
												</div>
											</Col>
											<Col md={4} className="text-center mb-4">
												<h5 className="text-light mb-3">Probability</h5>
												<CircularProgress
													value={result.probability || 0}
													color={
														result.prediction === 1 ? "#dc3545" : "#28a745"
													}
												/>
											</Col>
										</Row>
									</Card.Body>
								</Card>
							)}
						</Form>
					</Col>
				</Row>
			</Container>
		</div>
	)
}

export default App
