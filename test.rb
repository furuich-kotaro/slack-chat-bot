clients subscriptions.map { |subscription|
  stripe_customer = Stripe::Customer.retrieve(subscription.customer)
  {
    code: stripe_customer.metadata.try(:client_code),
    paymen_mehod: payment_method_mapping[subscription.collection_method],
    period_type: subscription&.plan&.interval,
    is_task_limit_active: true,
    task_limit: (subscription&.plan&.metadata.try(:operation_upper_limit) || 25_0000),
    old_payment_ended_at: Time.zone.at(subscription.current_period_end),
  }
}
