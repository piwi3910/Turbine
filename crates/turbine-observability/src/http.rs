//! Tower layers for every HTTP request: `x-request-id` echo/generation (recorded on the
//! request's tracing span) and bounded-label HTTP metrics.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::extract::MatchedPath;
use axum::http::{HeaderName, HeaderValue, Method, Request, Response};
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use tower::{Layer, Service};
use tracing::Instrument;

use crate::MetricsRegistry;

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Header carrying the request id in both directions.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");
const MAX_REQUEST_ID_LEN: usize = 128;
/// Route label for requests the router did not match; keeps raw paths out of labels.
const UNMATCHED_ROUTE: &str = "unmatched";

/// Request extension carrying the request's id (client-supplied or generated).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestIdExt(pub String);

/// A client id is kept when it is 1..=128 visible ASCII characters (0x21..=0x7E).
pub fn valid_request_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REQUEST_ID_LEN
        && value.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

/// Layer that echoes a valid client `x-request-id` or generates a UUIDv4, stores it as
/// [`RequestIdExt`], records it on the `http_request` span and sets it on the response.
pub fn request_id_layer() -> RequestIdLayer {
    RequestIdLayer
}

/// See [`request_id_layer`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestIdLayer;

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdService { inner }
    }
}

/// Service produced by [`RequestIdLayer`].
#[derive(Clone, Debug)]
pub struct RequestIdService<S> {
    inner: S,
}

/// The client's id when valid, otherwise a fresh UUIDv4, with its header value.
fn resolve_request_id(client: Option<&HeaderValue>) -> (String, HeaderValue) {
    if let Some(value) = client
        && let Ok(text) = value.to_str()
        && valid_request_id(text)
    {
        return (text.to_string(), value.clone());
    }
    let id = uuid::Uuid::new_v4().to_string();
    let header = HeaderValue::from_str(&id).expect("a hyphenated UUID is a valid header value");
    (id, header)
}

impl<S, B, R> Service<Request<B>> for RequestIdService<S>
where
    S: Service<Request<B>, Response = Response<R>>,
    S::Future: Send + 'static,
    S::Error: 'static,
    R: 'static,
{
    type Response = Response<R>;
    type Error = S::Error;
    type Future = BoxFuture<Result<Response<R>, S::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<B>) -> Self::Future {
        let (id, header) = resolve_request_id(req.headers().get(&REQUEST_ID_HEADER));
        req.headers_mut().insert(REQUEST_ID_HEADER, header.clone());
        let span = tracing::info_span!(
            "http_request",
            request_id = %id,
            method = %req.method(),
            path = %req.uri().path(),
        );
        req.extensions_mut().insert(RequestIdExt(id));
        let fut = span.in_scope(|| self.inner.call(req));
        Box::pin(
            async move {
                let mut resp = fut.await?;
                resp.headers_mut().insert(REQUEST_ID_HEADER, header);
                Ok(resp)
            }
            .instrument(span),
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RequestLabels {
    method: &'static str,
    route: String,
    status: u16,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct DurationLabels {
    method: &'static str,
    route: String,
}

/// 1 ms doubling to ~33 s.
fn duration_histogram() -> Histogram {
    Histogram::new(exponential_buckets(0.001, 2.0, 16))
}

/// `turbine_http_requests_total{method,route,status}` and
/// `turbine_http_request_duration_seconds{method,route}`.
#[derive(Clone, Debug)]
pub struct HttpMetrics {
    requests: Family<RequestLabels, Counter>,
    duration: Family<DurationLabels, Histogram, fn() -> Histogram>,
}

impl HttpMetrics {
    /// Register both HTTP metric families in `reg`.
    pub fn register(reg: &MetricsRegistry) -> Self {
        // prometheus-client appends `_total` to counters.
        let requests = reg.register(
            "turbine_http_requests",
            "HTTP requests by method, matched route template and status",
            Family::<RequestLabels, Counter>::default(),
        );
        let duration = reg.register(
            "turbine_http_request_duration_seconds",
            "HTTP request duration by method and matched route template",
            Family::<DurationLabels, Histogram, fn() -> Histogram>::new_with_constructor(
                duration_histogram,
            ),
        );
        HttpMetrics { requests, duration }
    }
}

/// Closed method label set; anything else is `OTHER`.
fn method_label(method: &Method) -> &'static str {
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::PATCH => "PATCH",
        _ => "OTHER",
    }
}

/// Layer recording [`HttpMetrics`] per request, labelled with the matched route template
/// (from Axum's [`MatchedPath`]) or `unmatched`. Must sit inside the router (`route_layer`
/// or `Router::layer`) so that `MatchedPath` is present.
pub fn http_metrics_layer(m: HttpMetrics) -> HttpMetricsLayer {
    HttpMetricsLayer { metrics: m }
}

/// See [`http_metrics_layer`].
#[derive(Clone, Debug)]
pub struct HttpMetricsLayer {
    metrics: HttpMetrics,
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HttpMetricsService {
            inner,
            metrics: self.metrics.clone(),
        }
    }
}

/// Service produced by [`HttpMetricsLayer`].
#[derive(Clone, Debug)]
pub struct HttpMetricsService<S> {
    inner: S,
    metrics: HttpMetrics,
}

impl<S, B, R> Service<Request<B>> for HttpMetricsService<S>
where
    S: Service<Request<B>, Response = Response<R>>,
    S::Future: Send + 'static,
    S::Error: 'static,
    R: 'static,
{
    type Response = Response<R>;
    type Error = S::Error;
    type Future = BoxFuture<Result<Response<R>, S::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        let method = method_label(req.method());
        // Route templates are a closed set fixed by the router; unmatched paths collapse
        // into one value so label cardinality cannot grow with traffic.
        let route = req
            .extensions()
            .get::<MatchedPath>()
            .map_or(UNMATCHED_ROUTE, MatchedPath::as_str)
            .to_string();
        let metrics = self.metrics.clone();
        let started = Instant::now();
        let fut = self.inner.call(req);
        Box::pin(async move {
            let resp = fut.await?;
            metrics
                .duration
                .get_or_create(&DurationLabels {
                    method,
                    route: route.clone(),
                })
                .observe(started.elapsed().as_secs_f64());
            metrics
                .requests
                .get_or_create(&RequestLabels {
                    method,
                    route,
                    status: resp.status().as_u16(),
                })
                .inc();
            Ok(resp)
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Method;

    use super::*;

    #[test]
    fn request_id_validation() {
        assert!(valid_request_id("abc-123"));
        assert!(valid_request_id(&"x".repeat(128)));
        assert!(!valid_request_id(&"x".repeat(129)));
        assert!(!valid_request_id(""));
        assert!(!valid_request_id("has space"));
        assert!(!valid_request_id("tab\there"));
        assert!(!valid_request_id("caf\u{e9}"));
        assert_eq!(method_label(&Method::GET), "GET");
        assert_eq!(method_label(&Method::from_bytes(b"BREW").unwrap()), "OTHER");
    }
}
